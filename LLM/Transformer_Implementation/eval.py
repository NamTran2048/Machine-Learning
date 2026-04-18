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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    model.eval()
    with torch.no_grad():
        #Retrieve latest model
        if (config['preload']):
            model_filename = get_weights_file_path(config, config['preload'])
            state = torch.load(model_filename, map_location=device)
            model.load_state_dict(state['model_state_dict'])

        #Encoder and decoder input/output
        encoder_input = get_encoder_input(text, tokenizer_src, tokenizer_tgt, text.size(-1)).to(device)
        encoder_output = model.encode(encoder_input.unsqueeze(0), None) #No masking, (1, seq)
        decoder_input = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64).to(device)
    
    

        #Inference loop

        while True:
            decoder_mask = get_decoder_mask(len(decoder_input)).to(device)

            decoder_output = model.decode(decoder_input.unsqueeze(0), encoder_output, None, decoder_mask)
            Proj_output = torch.softmax(model.projectionLayer(decoder_output), dim=-1) #(1,seq,vocab)
            output = Proj_output[0, -1, :]

            unk_id = tokenizer_tgt.token_to_id("[UNK]")
            output[unk_id] = -float('inf')
            next_token = torch.argmax(output)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)])
            tokens = [tokenizer_tgt.id_to_token(idx.item()) for idx in decoder_input]
            print(" ".join(tokens))

            if next_token == tokenizer_tgt.token_to_id("[EOS]"):
                break


if __name__ == '__main__':
    sentence = " I would rather sleep "
    config = get_config()
    translate(sentence, config)

    #I found joy by doing what I love . --> Mi sono trovato con gioia quello che amo .
    #I want to sleep tonight --> Voglio dormire nei campi .
    #Women should have the right to be independent and educated. --> D ’ altra parte , la donna può essere indipendente e educato .
    #I love food so much --> Ho bisogno di mangiare un tale che mi piace ?
    #I beg for your forgiveness, sir . --> Vi prego soltanto di perdonare , signore .