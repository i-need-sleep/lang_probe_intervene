import argparse
import os
import datetime
from copy import deepcopy

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, OPTForCausalLM

import models.probes as probes

import utils.data_utils as data_utils
import utils.globals as uglobals

def main():
    probe_type = 'linear'
    checkpoint_path = f'{uglobals.CHECKPOINTS_DIR}/linear_layer0.bin'
    layer_idx = 0
    debug = True

    # Device
    if torch.cuda.is_available() and not debug:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the OPT model
    tokenizer = AutoTokenizer.from_pretrained(uglobals.OPT_TOKENIZER_DIR)
    model = OPTForCausalLM.from_pretrained(uglobals.OPT_MODEL_DIR).to(device)

    text = 'The quick brown fox jumps over the lazy dog.'
    
    tokenized = tokenizer(text, return_tensors='pt').to(device)
    hidden_states1 = model(tokenized['input_ids'], tokenized['attention_mask'], output_hidden_states=True).hidden_states[0]
    tokenized['input_ids'][0, 0] = 996
    print(tokenized['input_ids'])

    tokenized['input_ids'][0, -1] = 996
    hidden_states2 = model(tokenized['input_ids'], tokenized['attention_mask'], output_hidden_states=True).hidden_states[0]

    print(tokenized['input_ids'])

    print(hidden_states1[0, 0, :])
    print(hidden_states2[0, -1, :])
    # Load the probe

    return

if __name__ == '__main__':
    main()