import torch
import torch.nn as nn
import math

"""

"""

class inputEmbedding(nn.Module):

    def __init__(self, d_model, vocab_size): #d_model is 512
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class positionalEncoder(nn.Module):

    def __init__(self, d_model, seq, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq,d_model)
        pos = torch.arange(0,seq).float().unsqueeze(1) #Size (seq, 1)
        div_term = torch.pow(10000,torch.arange(0, d_model, 2).float() / d_model)

        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)

        pe = pe.unsqueeze(0) # (1,seq,d_model). This is so we can batch in the future.
        self.register_buffer('pe', pe)


    def forward(self,x): #Our input is of size (batch, input, d_model)
        x = x + self.pe[:, 0:x.size(1)]
        return self.dropout(x)
    

class layerNorm(nn.Module):

    def __init__(self,d_model = 512):
        super().__init__()
        self.eps = 10**-6
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=2, keepdim=True) #Size (batch, input_seq, 1)
        std = x.std(dim=2, keepdim=True) #Size (batch, input_seq, 1)

        x = (x - mean)/torch.sqrt(std**2 + self.eps)
        return x * self.gamma + self.beta

class ffn(nn.Module):

    def __init__(self, d_model, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4*d_model)
        self.fc2 = nn.Linear(4*d_model, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))
    

class multiHeadAttention(nn.Module):

    def __init__(self, d_model, head, dropout): 
        super().__init__()
        self.d_model = d_model
        self.head = head
        assert d_model % head == 0 #Note, d_model % head == 0
        self.d_k = d_model // head
        self.dropout = nn.Dropout(dropout)

        self.wQ = nn.Linear(d_model,d_model)
        self.wK = nn.Linear(d_model,d_model)
        self.wV = nn.Linear(d_model,d_model)
        self.wO = nn.Linear(d_model,d_model)

    @staticmethod
    def attention(q, k, v, d_k, mask=None, dropout=None):
            Score = (q @ k.transpose(2,3)) / math.sqrt(d_k)
            if mask is not None:
                Score = Score.masked_fill(mask == 0, -1e8)
            Attention = Score.softmax(dim = -1)

            if dropout is not None:
                Attention = dropout(Attention)

            return (Attention @ v ), Attention

    def forward(self,q,k,v, mask = None):
        Q = self.wQ(q)
        K = self.wK(k)
        V = self.wV(v)

        Q = Q.view(Q.shape[0], Q.shape[1], self.head, self.d_k)
        Q = Q.transpose(1, 2)
        K = K.view(K.shape[0], K.shape[1], self.head, self.d_k)
        K = K.transpose(1, 2)
        V = V.view(V.shape[0], V.shape[1], self.head, self.d_k)
        V = V.transpose(1, 2)

        x, _ = self.attention(Q, K, V, self.d_k, mask, self.dropout)

        #Return back to our shape
        x = x.transpose(1,2).contiguous() 
        x = x.view(x.shape[0], -1 , self.d_model)

        return self.wO(x)
    
class residual(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = layerNorm()

    def forward(self, x, Sublayer):
        return x + self.dropout((Sublayer(self.norm(x))) * (1/(math.sqrt(12))))

class encoderBlock(nn.Module):

    def __init__(self, attention_block, FFN_block, dropout):
        super().__init__()
        self.attention_block = attention_block
        self.FFN_block = FFN_block
        self.residualConnection = nn.ModuleList([residual(dropout), residual(dropout)])

    def forward(self, x, mask):
        x = self.residualConnection[0](x, lambda x: self.attention_block(x,x,x, mask))
        x = self.residualConnection[1](x, self.FFN_block)
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    
class decoderBlock(nn.Module):

    def __init__(self, attention_block, cross_attention, FFN_block, dropout):
        super().__init__()
        self.attention_block = attention_block
        self.cross_block = cross_attention
        self.FFN_block = FFN_block
        self.norm = layerNorm()
        self.residualConnection = nn.ModuleList([residual(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.residualConnection[0](x, lambda x: self.attention_block(x,x,x, decoder_mask))
        x = self.residualConnection[1](x, lambda x: self.cross_block(x,encoder_output, encoder_output, encoder_mask))
        x = self.residualConnection[2](x, self.FFN_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = layerNorm()

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.Proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.Proj(x)

class Transformer(nn.Module): 

    def __init__(self, encoder, decoder, encoderEmbedding, decoderEmbedding, encoderPE, decoderPE, projection): #Each is instantiated already as objects
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoderEmbedding = encoderEmbedding
        self.decoderEmbedding = decoderEmbedding
        self.encoderPE = encoderPE
        self.decoderPE = decoderPE
        self.projection = projection

    def encode(self, x, mask):
        x = self.encoderEmbedding(x)
        x = self.encoderPE(x)
        return self.encoder(x, mask)

    def decode(self, x, encoderOutput, encoder_mask, decoder_mask):
        x = self.decoderEmbedding(x)
        x = self.decoderPE(x)
        return self.decoder(x, encoderOutput, encoder_mask, decoder_mask)
        
    def projectionLayer(self, x):
        return self.projection(x)
    


def Model(encoder_vocab_size, decoder_vocab_size, encoder_seq, decoder_seq, d_model, head = 8, N=12, dropout=0.1):
    #Embedding
    encoder_embedding = inputEmbedding(d_model, encoder_vocab_size)
    decoder_embedding = inputEmbedding(d_model, decoder_vocab_size)

    #Positional embedding
    encoder_positional = positionalEncoder(d_model, encoder_seq, dropout)
    decoder_positional = positionalEncoder(d_model, decoder_seq, dropout)

    #Encoder, Decoder block
    encoder_blocks = []
    for _ in range(N):
        MHA = multiHeadAttention(d_model, head, dropout)
        FFN = ffn(d_model, dropout)
        encoder_block = encoderBlock(MHA, FFN, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        MHA = multiHeadAttention(d_model, head, dropout)
        MHA_cross = multiHeadAttention(d_model, head, dropout)
        FFN = ffn(d_model, dropout)
        decoder_block = decoderBlock(MHA, MHA_cross, FFN, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Projection
    projection = ProjectionLayer(d_model, decoder_vocab_size)

    #Transformer
    transformer = Transformer(encoder, decoder, encoder_embedding, decoder_embedding, encoder_positional, decoder_positional, projection)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer






