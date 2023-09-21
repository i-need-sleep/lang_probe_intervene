import argparse
import os
import datetime
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, OPTForCausalLM
from tqdm import tqdm

import models.probes as probes

import utils.data_utils as data_utils
import utils.globals as uglobals

def main():
    probe_type = 'linear'
    checkpoint_path = f'{uglobals.CHECKPOINTS_DIR}/linear_layer0.bin'
    layer_idx = 0
    debug = True
    batch_size = 32

    # Device
    if torch.cuda.is_available() and not debug:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the OPT model
    tokenizer = AutoTokenizer.from_pretrained(uglobals.OPT_TOKENIZER_DIR)
    opt = OPTForCausalLM.from_pretrained(uglobals.OPT_MODEL_DIR).to(device)

    # for text in [
    #     'wins pigs at overdue grandparents and',
    #     'looks farms at upcoming photos and'
    #     ]:
    #     tokenized = tokenizer(text, return_tensors='pt').to(device)
    #     print(tokenized)
    # exit()

    # Load the probe
    if probe_type == 'linear':
        probe = probes.LinearProbe().to(device)
    elif probe_type == 'mlp':
        probe = probes.MLPProbe().to(device)
    else:
        raise ValueError(f'Unknown probe type: {probe_type}')
    checkpoint = torch.load(checkpoint_path)
    probe.load_state_dict(checkpoint['model_state_dict'])

    # Data
    # train_loader = data_utils.make_hidden_states_loader(f'{uglobals.TRAINING_DIR}/train_{layer_idx}.pt', layer_idx, batch_size, shuffle=True)
    # dev_loader = data_utils.make_hidden_states_loader(f'{uglobals.TRAINING_DIR}/val_{layer_idx}.pt', layer_idx, batch_size, shuffle=False)

    # for batch in dev_loader:
    #     hidden_state = batch['hidden_state'].to(device)
    #     target = batch['singular'].float().to(device).reshape(-1)

    #     logits = probe(hidden_state).reshape(-1)

    #     print(torch.round(torch.sigmoid(logits)))
    #     print(target)
    #     exit()

    train_set = data_utils.HiddenStatesDataset(0, f'{uglobals.TRAINING_DIR}/train_{layer_idx}.pt')
    dev_set = data_utils.HiddenStatesDataset(0, f'{uglobals.TRAINING_DIR}/val_{layer_idx}.pt')

    n_overlap = 0
    n_not_overlap = 0
    n_all = 0
    for train_idx, train_line in tqdm(enumerate(train_set[:1000])):
        n_all += 1
        train_hidden_state = train_line['hidden_state']
        for dev_idx, dev_line in enumerate(dev_set):
            dev_hidden_state = dev_line['hidden_state']
            if torch.sum(train_hidden_state - dev_hidden_state) == 0 and train_line['singular'] == dev_line['singular']:
                n_overlap += 1
            if torch.sum(train_hidden_state - dev_hidden_state) == 0 and train_line['singular'] != dev_line['singular']:
                n_not_overlap += 1
                
        if train_idx % 100 == 0:
            print(n_overlap)
            print(n_not_overlap)
            print(n_all)

    return

if __name__ == '__main__':
    main()