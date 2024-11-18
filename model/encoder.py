import torch
import torch.nn as nn
from model.positional_encoding import PositionalEncoding
from model.attention import SingleHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, dim, dim_ff):
        super().__init__()
        self.self_attention = SingleHeadAttention(dim)
        self.norm_attn = nn.LayerNorm(dim)
        # Fully connected feed forward network
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        # Dropout for regularization of 
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        x2, _ = self.self_attention(x, x, x, mask)
        x = self.norm_attn(x + self.dropout(x2))
        # Feed-forward network and residual connection
        x2 = self.fc2(torch.relu(self.fc1(x)))
        x = self.norm_ffn(x + self.dropout(x2))
        return x    
        

class Encoder(nn.Module):
    def __init__(self, dim, num_layers, dim_ff, max_seq_len, patch_size=7):  # 7x7 patches
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2  # For MNIST 28x28 images
        patch_dim = patch_size * patch_size  # 49 for 7x7 patches
        
        # Patch embedding layer
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.positional_encoding = PositionalEncoding(dim, self.num_patches)
        self.layers = nn.ModuleList([EncoderLayer(dim, dim_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, input_seq):
        # input_seq shape: [batch_size, 784]
        batch_size = input_seq.shape[0]
        
        # Reshape into image
        x = input_seq.view(batch_size, 1, 28, 28)
        
        # Extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)
        
        # Embed patches
        x = self.patch_embedding(patches)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)
            
        return self.norm(x)
