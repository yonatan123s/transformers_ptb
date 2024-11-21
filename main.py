import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import math
from data import PTBDataset, batchify
from model import TransformerModel
from train import train, evaluate
import os

def main():
    parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model with Lipschitz Regularization')
    parser.add_argument('--data', type=str, default='ptbdataset',
                        help='location of the data corpus')
    parser.add_argument('--embed_size', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=2,
                        help='number of heads in the Transformer encoder')
    parser.add_argument('--lr', type=float, default=5.0,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Load data
    train_dataset = PTBDataset(os.path.join(args.data, 'train.txt'))
    vocab = train_dataset
    valid_dataset = PTBDataset(os.path.join(args.data, 'valid.txt'), vocab)
    test_dataset = PTBDataset(os.path.join(args.data, 'test.txt'), vocab)

    train_data = torch.tensor(train_dataset.data, dtype=torch.long)
    val_data = torch.tensor(valid_dataset.data, dtype=torch.long)
    test_data = torch.tensor(test_dataset.data, dtype=torch.long)

    train_data = batchify(train_data, args.batch_size, device)
    val_data = batchify(val_data, args.batch_size, device)
    test_data = batchify(test_data, args.batch_size, device)

    ntokens = len(vocab.word2idx)

    model = TransformerModel(ntokens, args.embed_size, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = None

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train(model, train_data, criterion, optimizer, scheduler, epoch, args.bptt, device)
        train_ppl = math.exp(train_loss)
        val_loss = evaluate(model, val_data, criterion, args.bptt, device)
        val_ppl = math.exp(val_loss)
        test_loss = evaluate(model, test_data, criterion, args.bptt, device)
        test_ppl = math.exp(test_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'train loss {:5.2f} | train ppl {:8.2f} | '
              'valid loss {:5.2f} | valid ppl {:8.2f} | '
              'test loss {:5.2f} | test ppl {:8.2f}'.format(
            epoch, (time.time() - epoch_start_time),
            train_loss, train_ppl,
            val_loss, val_ppl,
            test_loss, test_ppl))
        print('-' * 89)

        if not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            with open('model.pt', 'wb') as f:
                torch.save(model.state_dict(), f)
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            scheduler.step()

    # Load the best saved model.
    with open('model.pt', 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Run on test data.
    test_loss = evaluate(model, test_data, criterion, args.bptt, device)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, test_ppl))
    print('=' * 89)

if __name__ == '__main__':
    main()
