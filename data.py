import torch
from torch.utils.data import Dataset
from collections import Counter
import os

class PTBDataset(Dataset):
    def __init__(self, data_path, vocab=None):
        with open(data_path, 'r') as f:
            text = f.read()

        self.tokens = text.replace('\n', '<eos>').split()
        if vocab is None:
            self.build_vocab()
        else:
            self.word2idx = vocab.word2idx
            self.idx2word = vocab.idx2word

        self.data = [self.word2idx.get(token, self.word2idx['<unk>']) for token in self.tokens]

    def build_vocab(self):
        counter = Counter(self.tokens)
        counter['<unk>'] = 0  # Ensure '<unk>' is in vocab
        vocab = sorted(counter.items(), key=lambda x: -x[1])
        self.idx2word = [word for word, freq in vocab]
        self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

    def __len__(self):
        return len(self.data)

def batchify(data, batch_size, device):
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)
