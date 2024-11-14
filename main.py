# main.py

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import PTBDataset
from model import GPTModel
from utils import save_checkpoint, load_checkpoint, compute_perplexity

def train(args):
    # Load datasets
    train_dataset = PTBDataset(f'{args.data_dir}/train.txt', seq_length=args.seq_length)
    vocab = train_dataset.word2idx
    valid_dataset = PTBDataset(f'{args.data_dir}/valid.txt', vocab=vocab, seq_length=args.seq_length)
    test_dataset = PTBDataset(f'{args.data_dir}/test.txt', vocab=vocab, seq_length=args.seq_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GPTModel(
        vocab_size=len(vocab),
        embed_size=args.embed_size,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_seq_length=args.seq_length,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_ppl = float('inf')

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            # In the training loop, after loading inputs
            assert torch.min(inputs) >= 0, "Input contains negative indices."
            assert torch.max(inputs) < len(vocab), "Input contains indices larger than vocab size."

            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # outputs shape: (batch_size, seq_length, vocab_size)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_ppl = compute_perplexity(total_loss, len(train_loader))

        # Validation
        valid_ppl = evaluate(model, valid_loader, criterion, device)

        # Testing
        test_ppl = evaluate(model, test_loader, criterion, device)

        print(f'Epoch {epoch}: Train PPL: {train_ppl:.4f}, Valid PPL: {valid_ppl:.4f}, Test PPL: {test_ppl:.4f}')

        # Save best model
        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            save_checkpoint(model, args.cp_path)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
    ppl = compute_perplexity(total_loss, len(data_loader))
    return ppl

def test(args):
    # Load vocab
    train_dataset = PTBDataset(f'{args.data_dir}/train.txt', seq_length=args.seq_length)
    vocab = train_dataset.word2idx
    idx2word = train_dataset.idx2word

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GPTModel(
        vocab_size=len(vocab),
        embed_size=args.embed_size,
        num_heads=args.num_heads,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        max_seq_length=args.seq_length,
        dropout=args.dropout
    ).to(device)

    load_checkpoint(model, args.cp_path)
    model.eval()

    print('Enter a sentence to generate the next word (type "quit" to exit):')

    while True:
        input_text = input('Input: ')
        if input_text.lower() == 'quit':
            break
        tokens = input_text.strip().split()
        tokens_idx = [vocab.get(token, vocab['<unk>']) for token in tokens]
        inputs = torch.tensor(tokens_idx, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            next_token_logits = outputs[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            next_token = idx2word[next_token_id]
            print(f'Next word: {next_token}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='ptbdataset')
    parser.add_argument('--seq_length', type=int, default=32)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--cp_path', type=str, default='best_model.pt')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)

if __name__ == '__main__':
    main()
