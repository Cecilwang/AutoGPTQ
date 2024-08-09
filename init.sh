pip install virtualenv
virtualenv ../vir/autogptq
source ../vir/autogptq/bin/activate
pip install numpy gekko pandas torch transformers datasets
pip install -vvv --no-build-isolation -e .