from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epoch": 20,
        "seq_len": 250,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "10",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel", #Allow resume if model crashes
    }

def get_weights_file_path(config, epoch):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)