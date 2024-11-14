# model.py

import torch
import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers, max_seq_length, dropout=0.1):
        super(GPTModel, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_seq_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, hidden_size, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.positional_embedding(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.fc_out(x)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embed_size)
        seq_length = x.size(1)
        attn_output, _ = self.attention(
            x, x, x,
            attn_mask=self._generate_subsequent_mask(seq_length).to(x.device)
        )
        x = x + self.dropout(attn_output)
        x = self.layer_norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm2(x)
        return x

    def _generate_subsequent_mask(self, size):
        mask = torch.triu(torch.full((size, size), float('-inf')), diagonal=1)
        return mask  # Shape: (seq_length, seq_length)

