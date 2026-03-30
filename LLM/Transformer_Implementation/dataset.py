import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TransformerDataset(Dataset):

    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq):
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq = seq

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang] #Select text in english
        tgt_text = src_target_pair['translation'][self.tgt_lang] #Select text in Spanish

        encoder_token = self.tokenizer_src.encode(src_text).ids
        decoder_token = self.tokenizer_tgt.encode(tgt_text).ids

        encoder_padding_token = self.seq - len(encoder_token) - 2 #SOS and EOS
        decoder_padding_token = self.seq - len(decoder_token) - 1 #Only SOS

        if encoder_padding_token < 0 or decoder_padding_token < 0:
            raise ValueError("Sequence length too short")
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()]* encoder_padding_token, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_token, dtype=torch.int64),
                torch.tensor([self.pad_token.item()]* decoder_padding_token, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(decoder_token, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()]* decoder_padding_token, dtype=torch.int64)
            ]

        )

        return {
            "encoder input": encoder_input, #(seq_length)
            "decoder input": decoder_input, 
            "encoder mask": (encoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0).int(), #(1,1, seq)
            "decoder mask": (decoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0).int() & decoder_mask(decoder_input.size(0)), #(1,1, seq)
            "label": label,
            "source text": src_text,
            "target text": tgt_text,
        }
    
def decoder_mask(seq):
    mask = torch.triu(torch.ones(1, seq,seq), diagonal=1).type(torch.int)
    return mask == 0.
    


