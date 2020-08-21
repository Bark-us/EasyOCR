mkdir -p model_store
torch-model-archiver --model-name craft --version 1.0 --model-file craft.py --serialized-file compat_models/craft.pth --extra-files craft_utils.py --export-path model_store --handler craft_handler.py -f
torch-model-archiver --model-name text --version 1.0 --model-file text.py --serialized-file compat_models/text.pth --extra-files text_utils.py --export-path model_store --handler text_handler.py -f