import torch
import torch.nn as nn
from model import Model
from train import get_sentence, buildTokenizer, get_model, buildTokenizer
from config import get_config, get_weights_file_path
from datasets import load_dataset

def get_encoder_input(text, tokenizer_src, tokenizer_tgt, seq):
    sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    encoder_input = torch.cat(
        [
            sos_token,
            torch.tensor(text, dtype=torch.int64),
            eos_token,
        ]
    )

    return encoder_input

def get_decoder_mask(seq):
    mask = torch.triu(torch.ones(1, seq,seq), diagonal=1).type(torch.int)
    return mask == 0.

def translate(sentence, config):
    dataset = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}', split = "train")
    tokenizer_src = buildTokenizer(config, dataset, config['lang_src']) 
    tokenizer_tgt = buildTokenizer(config, dataset, config['lang_tgt'])

    text = torch.tensor(tokenizer_src.encode(sentence).ids)
    print(text)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())

    #Retrieve latest model
    if (config['preload']):
        model_filename = get_weights_file_path(config, config['preload'])
        state = torch.load(model_filename)

    #Encoder and decoder input/output
    encoder_input = get_encoder_input(text, tokenizer_src, tokenizer_tgt, text.size(-1))
    encoder_output = model.encode(encoder_input.unsqueeze(0), None) #No masking, (1, seq)
    decoder_input = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    
    

    #Inference loop
    while (tokenizer_tgt.token_to_id("[EOS]") not in decoder_input):
        if len(decoder_input) > 1:
            decoder_mask = get_decoder_mask(len(decoder_input))
        else:
            decoder_mask = None

        decoder_output = model.decode(decoder_input.unsqueeze(0), encoder_output, None, decoder_mask)
        Proj_output = model.projectionLayer(decoder_output) #(1,seq,vocab)
        output = torch.softmax(Proj_output, dim = -1).squeeze()

        next_token = torch.argmax(output[-1])
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)])
        tokens = [tokenizer_tgt.id_to_token(idx.item()) for idx in decoder_input]
        print(tokens)

if __name__ == '__main__':
    sentence = "I like to eat alot"
    config = get_config()
    translate(sentence, config)