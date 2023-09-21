import argparse
import os
import datetime
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter

import models.probes as probes

import utils.data_utils as data_utils
import utils.globals as uglobals

def main(args, layer_idx):
    # Device
    torch.manual_seed(21)
    if torch.cuda.is_available() and not args.debug:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Setup Tensorboard
    date_str = str(datetime.datetime.now())[:-7].replace(':','-')
    writer = SummaryWriter(log_dir=f'{uglobals.RESULTS_DIR}/runs/{args.name}/batch_size={args.batch_size}, Adam_lr={args.lr}/{date_str}', comment=args)

    # Data
    train_loader = data_utils.make_hidden_states_loader(f'{uglobals.TRAINING_DIR}/train_{layer_idx}.pt', layer_idx, batch_size=args.batch_size, shuffle=True)
    dev_loader = data_utils.make_hidden_states_loader(f'{uglobals.TRAINING_DIR}/val_{layer_idx}.pt', layer_idx, batch_size=args.batch_size, shuffle=False)

    # Model
    if args.probe_type == 'linear':
        model = probes.LinearProbe().to(device)
    elif args.probe_type == 'mlp':
        model = probes.MLPProbe().to(device)
    else:
        raise ValueError(f'Unknown probe type: {args.probe_type}')

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training
    n_iter = 0
    n_prev_iter = 0
    running_loss = 0
    best_dev_loss = float('inf')
    for epoch in range(args.n_epoches):
        #  Train
        for batch_idx, batch in enumerate(train_loader):
            if args.debug and batch_idx > 3:
                break
            loss = train_step(batch, model, optimizer, criterion, device)
            n_iter += 1
            writer.add_scalar('Loss/train', loss, n_iter)
            running_loss += loss.detach()

        # Batch loss
        print(f'Epoch {epoch} done. Loss: {running_loss/(n_iter-n_prev_iter)}')
        writer.add_scalar('Loss/train_avg', running_loss/(n_iter-n_prev_iter), n_iter)
        n_prev_iter = n_iter
        running_loss = 0

        # Dev
        dev_loss = 0
        n_hits = 0
        n_total = 0
        for batch_idx, batch in enumerate(dev_loader):
            if args.debug and batch_idx > 3:
                break
            
            dev_loss_iter, n_hits_iter = dev_step(batch, model, criterion, device)
            dev_loss += dev_loss_iter.detach()
            n_hits += n_hits_iter
            n_total += len(batch['hidden_state'])

        dev_loss = dev_loss / len(dev_loader)
        print(f'Dev loss: {dev_loss}')
        writer.add_scalar('loss/dev', dev_loss, n_iter)

        dev_acc = n_hits / n_total
        print(f'Dev accuracy: {dev_acc}')
        writer.add_scalar('accuracy/dev', dev_acc, n_iter)

        # Save
        if dev_loss < best_dev_loss and not args.debug:
            best_dev_loss = dev_loss
            print(f'Best dev loss: {best_dev_loss}')
            try:
                os.makedirs(f'{uglobals.CHECKPOINTS_DIR}/{args.name_root}')
            except FileExistsError:
                pass
            try:
                os.makedirs(f'{uglobals.CHECKPOINTS_DIR}/{args.name_root}/{args.name}')
            except FileExistsError:
                pass
            save_dir = f'{uglobals.CHECKPOINTS_DIR}/{args.name_root}/{args.name}/lr{args.lr}.bin'
            print(f'Saving at: {save_dir}')
            torch.save({
                'epoch': epoch,
                'step': n_iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, save_dir)

    return

def train_step(batch, model, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    # Unpack the batch
    hidden_state = batch['hidden_state'].to(device)
    target = batch['singular'].float().to(device).reshape(-1)

    # Forward
    logits = model(hidden_state).reshape(-1)

    # Loss
    loss = criterion(logits, target)

    # Backward
    loss.backward()
    optimizer.step()

    return loss

def dev_step(batch, model, criterion, device):
    model.eval()

    with torch.no_grad():
        # Unpack the batch
        hidden_state = batch['hidden_state'].to(device)
        target = batch['singular'].float().to(device).reshape(-1)

        # Forward
        logits = model(hidden_state).reshape(-1)

        # Loss
        loss = criterion(logits, target)

        # Get accuracy
        n_hits = torch.sum(torch.round(torch.sigmoid(logits)) == target).item()

        return loss, n_hits

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name_root', default='unnamed')
    parser.add_argument('--debug', action='store_true')

    # Formulation
    parser.add_argument('--probe_type', type=str) # linear, mlp

    # Training
    parser.add_argument('--n_epoches', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', default=3e-5, type=float)

    args = parser.parse_args()

    if args.debug:
        args.n_epoches = 1
        args.batch_size = 8
    print(args)
    
    for layer_idx in range(25):
        print(f'-----Probing layer {layer_idx}-----')
        args.name = f'{args.name_root}_layer{layer_idx}'
        main(args, layer_idx)