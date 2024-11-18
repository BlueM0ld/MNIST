import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, dim_model, num_encoder_layers, num_decoder_layers, dim_ff, max_len, output_dim, patch_size=7):
        super().__init__()
        
        # Create encoder with patch embedding
        self.encoder = Encoder(dim_model, num_encoder_layers, dim_ff, max_len, patch_size)
        self.decoder = Decoder(dim_model, num_decoder_layers, dim_ff, max_len)
        
        # Add output projection layer for classification
        self.output_projection = nn.Linear(dim_model, output_dim)
        
        # Initialize weights
        self._init_parameters()
    
    def _init_parameters(self):
        # Initialize parameters with Xavier uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src_seq):
        # For MNIST classification, we don't need a target sequence
        # Shape: [batch_size, seq_len] -> [batch_size, seq_len, dim_model]
        encoder_output = self.encoder(src_seq)
        
        pooled_output = encoder_output.mean(dim=1)  # [batch_size, dim_model]
        
        # Project to output dimension
        logits = self.output_projection(pooled_output)  # [batch_size, output_dim]
        
        return logits
