import torch
import time
import math
from torch import nn

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def train(model, train_data, criterion, optimizer, scheduler, epoch, bptt, device):
    model.train()
    total_loss = 0.
    total_epoch_loss = 0.
    start_time = time.time()
    ntokens = model.decoder.out_features
    src_mask = None
    num_batches = train_data.size(0) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        if src_mask is None or src_mask.size(0) != len(data):
            src_mask = model._generate_square_subsequent_mask(len(data)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        total_epoch_loss += loss.item()
        if batch % 200 == 0 and batch > 0:
            cur_loss = total_loss / 200
            ppl = math.exp(cur_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, num_batches, scheduler.get_last_lr()[0],
                    elapsed * 1000 / 200,
                    cur_loss, ppl))
            total_loss = 0
            start_time = time.time()
    scheduler.step()
    return total_epoch_loss / num_batches

def evaluate(model, eval_data, criterion, bptt, device):
    model.eval()
    total_loss = 0.
    ntokens = model.decoder.out_features
    src_mask = None
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            data = data.to(device)
            targets = targets.to(device)
            if src_mask is None or src_mask.size(0) != len(data):
                src_mask = model._generate_square_subsequent_mask(len(data)).to(device)
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)
