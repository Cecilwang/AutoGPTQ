from argparse import ArgumentParser
import random
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from auto_gptq.modeling import BaseGPTQForCausalLM
from auto_gptq import BaseQuantizeConfig

def get_c4(dataset, nsamples, seqlen, tokenizer, seed):
    data_files = {'train': []}
    if 'en' in dataset or dataset == 'c4':
        data_files['train'] += ['en/c4-train.00000-of-01024.json.gz']
    if 'ja' in dataset or dataset == 'c4':
        data_files['train'] += ['multilingual/c4-ja.tfrecord-00000-of-01024.json.gz']
    traindata = load_dataset(
        'allenai/c4', data_files=data_files, split='train'
    )
    # data_files = {'validation': []}
    # if 'en' in dataset or dataset == 'c4':
    #     data_files['validation'] += ['en/c4-validation.00000-of-00008.json.gz']
    # if 'ja' in dataset or dataset == 'c4':
    #     data_files['validation'] += ['multilingual/c4-ja-validation.tfrecord-00000-of-00008.json.gz']
    # valdata = load_dataset(
    #     'allenai/c4', data_files=data_files, split='validation'
    # )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        trainloader.append({"input_ids": inp, "attention_mask": attention_mask})
    print(f"trainloader lenght {len(trainloader)}")

    return trainloader


def get_loader(dataset, nsamples, seqlen, tokenizer, seed):
    if 'c4' in dataset:
        return get_c4(dataset, nsamples, seqlen, tokenizer, seed)
    else:
        raise NotImplementedError()


class PLaMoGPTQForCausalLM(BaseGPTQForCausalLM):
    # chained attribute name of transformer layer block
    layers_block_name = "model.layers.layers"
    # chained attribute names of other nn modules that in the same level as the transformer layer block
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    # chained attribute names of linear layers in transformer layer module
    # normally, there are four sub lists, for each one the modules in it can be seen as one operation,
    # and the order should be the order when they are truly executed, in this case (and usually in most cases),
    # they are: attention q_k_v projection, attention output projection, MLP project input, MLP project output
    inside_layer_modules = [
        ["self_attn.qkv_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_up_proj"],
        ["mlp.down_proj"]
    ]


def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--quantized_model_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="group size, -1 means no grouping or full rank",
    )
    parser.add_argument("--desc_act", action="store_true", help="whether to quantize with desc_act")
    parser.add_argument(
        "--dataset",
        type=str,
        default="c4",
        choices=["c4", "c4-en", "c4-ja"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="how many samples will be used to quantize model",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument("--fast_tokenizer", action="store_true", help="whether use fast tokenizer")
    parser.add_argument(
        "--use_triton",
        action="store_true",
        help="whether use triton to speedup at inference",
    )
    parser.add_argument(
        "--per_gpu_max_memory",
        type=int,
        default=None,
        help="max memory used to load model per gpu",
    )
    parser.add_argument(
        "--cpu_max_memory",
        type=int,
        default=None,
        help="max memory used to offload model to cpu",
    )
    parser.add_argument(
        "--quant_batch_size",
        type=int,
        default=1,
        help="examples batch size for quantization",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="whether to trust remote code when loading model",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=4096,
    )
    args = parser.parse_args()

    max_memory = {}
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_dir,
        use_fast=args.fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    model = PLaMoGPTQForCausalLM.from_pretrained(
        args.pretrained_model_dir,
        quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=args.group_size, desc_act=args.desc_act),
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code,
    )

    traindataset = get_loader(args.dataset, args.num_samples, args.seqlen, tokenizer, args.seed)

    start = time.time()
    model.quantize(
        traindataset,
        batch_size=args.quant_batch_size,
        use_triton=args.use_triton,
        autotune_warmup_after_quantized=args.use_triton,
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    if args.quantized_model_dir:
        # save quantized model using safetensors
        model.save_quantized(args.quantized_model_dir, use_safetensors=True)
        tokenizer.save_pretrained(args.quantized_model_dir)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
