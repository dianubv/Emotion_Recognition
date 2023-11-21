import torch
import torch.nn as nn

from math import log

class PositionalEncoding(nn.Module):
    """ Give a sense of position to the input tensor"""
    def __init__(self, d_model=16, max_len=8000) -> None:
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', self.generate_encoding(d_model, max_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assuming x has shape (seq_len, batch_size, d_model)
        seq_len, batch_size, _ = x.size()
        x = x + self.pe[:seq_len, :]  # Add positional encoding to the input tensor
        return x

    def generate_encoding(self, d_model : int, max_len : int) -> torch.Tensor:
        """ Generate positional encoding for transformer """
        pe = torch.zeros(max_len, d_model)      # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # Generate positions from 0 to (max_len - 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))  # generat a tensor, containing different frequencies of sin and coss function, with a shape (d_model / 2) containing values of 10000 ^ (2i / d_model)
        
        #Calculate sine and cosine values for each position and feature dimension
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(1)  # shape : (max_len, 1, d_model)


class ConvTransformer(nn.Module):
    def __init__(self, conv_channels=8, conv_kernel_size=3, d_model=16, nhead=1, num_layers=1, dim_feedforward=64, output_dim=1) -> None:
        super(ConvTransformer, self).__init__()
        
        # Conv1D Layers
        self.conv1 = nn.Conv1d(1, conv_channels, conv_kernel_size, stride=1)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2)
        
        # Transformer Layers
        self.embedding = nn.Linear(conv_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model)  # Assuming you have defined PositionalEncoding
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final FC Layer
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Conv1D Layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Reshape for Transformer
        batch_size, channels, seq_len = x.size()
        x = x.view(batch_size, seq_len, channels)  # Reshape to (batch_size, seq_len, channels)
        
        # Transformer Layers
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        x = self.global_avg_pool(x)
        x = x.squeeze(2)  # (batch_size, d_model)
        
        # Final FC Layer
        x = self.fc(x)
        return x
