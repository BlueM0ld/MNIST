import torch
import torch.nn as nn
from model.attention import SingleHeadAttention
from model.positional_encoding import PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, dim, dim_ff):
        super().__init__()
        self.self_attention = SingleHeadAttention(dim)
        self.cross_attention = SingleHeadAttention(dim)
        self.norm_self_attn = nn.LayerNorm(dim)
        self.norm_cross_attn = nn.LayerNorm(dim)
        
        # Fully connected feed forward network
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self attention
        x2, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm_self_attn(x + self.dropout(x2))
        
        # Cross attention
        x2, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm_cross_attn(x + self.dropout(x2))
        
        # Feed-forward network and residual connection
        x2 = self.fc2(torch.relu(self.fc1(x)))
        x = self.norm_ffn(x + self.dropout(x2))
        return x

class Decoder(nn.Module):
    def __init__(self, dim, num_layers, dim_ff, max_seq_len):
        super().__init__()
        self.positional_encoding = PositionalEncoding(dim, max_seq_len)
        self.layers = nn.ModuleList([DecoderLayer(dim, dim_ff) for _ in range(num_layers)])
        
        # Need to normalize the output of the last layer for stability
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, tgt_seq, encoder_output, src_mask=None, tgt_mask=None):
        x = self.positional_encoding(tgt_seq)
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
