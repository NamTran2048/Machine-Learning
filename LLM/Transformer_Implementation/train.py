import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel # Actually tokenize
from tokenizers.trainers import WordLevelTrainer #Create Vocab based on the sentences on the dataset
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import TransformerDataset, decoder_mask

from model import Model
from config import get_weights_file_path, get_config

def get_sentence(dataset, language):
    for item in dataset:
        yield item['translation'][language]

def buildTokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token = '[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 1)
        tokenizer.train_from_iterator(get_sentence(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_data(config):
    dataset = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = "train")

    #Creates tokenizer from function buildTokenizer
    tokenizer_src = buildTokenizer(config, dataset, config['lang_src']) 
    tokenizer_tgt = buildTokenizer(config, dataset, config['lang_tgt'])

    #Keep 90% for training, 10% for validation

    dataset_train_size = int(0.9 * len(dataset))
    dataset_eval_size = len(dataset) - dataset_train_size
    dataset_train_raw, dataset_eval_raw = random_split(dataset, [dataset_train_size, dataset_eval_size])


    #Actual dataset
    dataset_train = TransformerDataset(dataset_train_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])
    dataset_eval = TransformerDataset(dataset_eval_raw, tokenizer_src, tokenizer_tgt, config["lang_src"], config["lang_tgt"], config["seq_len"])

    #Dataloader
    train_dataloader = DataLoader(dataset_train, batch_size = config['batch_size'], shuffle = True)
    eval_dataloader = DataLoader(dataset_eval, batch_size = 1, shuffle = True)

    return train_dataloader, eval_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, encoder_vocab_size, decoder_vocab_size):
    model = Model(encoder_vocab_size, decoder_vocab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, eval_dataloader, tokenizer_src, tokenizer_tgt = get_data(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr= 0.5, eps=1e-9, weight_decay=1e-2)

    def lr_lambda(step):
        step = max(step, 1)
        warmup_steps = 10000
        return (config['d_model'] ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_function = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1) #Label_smoothing is activated

    initial_epoch = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])

    for epoch in range(initial_epoch, config["num_epoch"]):
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            encoder_input = batch['encoder input'].to(device) #(B,seq)
            decoder_input = batch['decoder input'].to(device) #(B,seq)
            encoder_mask = batch['encoder mask'].to(device) #(B,1,1,seq)
            decoder_mask = batch['decoder mask'].to(device)#(B,1,seq,seq)
            label = batch['label'].to(device) #(B, seq)

            #Transformer actions

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
            Proj_output = model.projectionLayer(decoder_output) #(B, seq, vocab_size)

            loss = loss_function(Proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1) ) #Convert it to (B * seq, vocab_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #Weight clipping
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, model_filename)

if __name__ == '__main__':
    config = get_config()
    train_model(config)

