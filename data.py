# data.py

import torch
from torch.utils.data import Dataset

class PTBDataset(Dataset):
    def __init__(self, file_path, vocab=None, seq_length=32):
        with open(file_path, 'r') as f:
            text = f.read()
        self.tokens = text.split()
        self.seq_length = seq_length

        if vocab is None:
            self.build_vocab()
            self.data = [self.word2idx.get(token, self.word2idx['<unk>']) for token in self.tokens]
            unknown_count = 0
            for token in self.tokens:
                idx = self.word2idx.get(token, self.word2idx['<unk>'])
                if idx == self.word2idx['<unk>']:
                    unknown_count += 1
                self.data.append(idx)

            print(f"Number of unknown tokens: {unknown_count} out of {len(self.tokens)} tokens")

        else:
            self.word2idx = vocab
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
            self.data = [self.word2idx.get(token, self.word2idx['<unk>']) for token in self.tokens]

    def build_vocab(self):
        tokens_set = set(self.tokens)
        tokens_set.discard('<pad>')
        tokens_set.discard('<unk>')
        vocab = ['<pad>', '<unk>'] + sorted(tokens_set)
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def __len__(self):
        return (len(self.data) - self.seq_length - 1) // self.seq_length + 1

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length
        if end_idx + 1 > len(self.data):
            end_idx = len(self.data) - 2  # Adjust to prevent out-of-bounds
            start_idx = end_idx - self.seq_length  # Ensure sequence length remains consistent
        x = torch.tensor(self.data[start_idx:end_idx], dtype=torch.long)
        y = torch.tensor(self.data[start_idx + 1:end_idx + 1], dtype=torch.long)
        return x, y

