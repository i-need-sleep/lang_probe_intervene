import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from . import globals as uglobals

class HiddenStatesDataset(Dataset):
    def __init__(self, layer, load_path=None):
        self.layer = layer # [0, 24]
        self.metadata = pd.read_csv(f'{uglobals.COLORLESS_GREEN_PATH}', sep='\t', header=0)
        self.tokenizer = AutoTokenizer.from_pretrained(uglobals.OPT_TOKENIZER_DIR)

        if load_path != None:
            data = torch.load(load_path)
            self.data = self.slice_timesteps(data) # {tensor(hidden_size), singular}

    def slice_timesteps(self, data):
        # list of tensors [(seq_len, hidden_size)] to list of tensors [(hidden_size)]
        out = []
        for line in data.values():
            hidden_states = line['hidden_state']
            singular = line['singular']
            for i in range(hidden_states.shape[0]):
                out.append({
                    'hidden_state': hidden_states[i, :],
                    'singular': singular
                })
        return out

    def make_splits(self, val_ratio=0.1):
        data = self.prep_data() # {sent_idx, [hidden_states]}

        # Split into train, val
        indices = list(data.keys())
        random.shuffle(indices)
        val_size = int(val_ratio * len(indices))
        val_indices = indices[: val_size]
        
        train_data = {}
        val_data = {}
        for sent_idx in data.keys():
            if sent_idx in val_indices:
                val_data[sent_idx] = data[sent_idx]
            else:
                train_data[sent_idx] = data[sent_idx]

        torch.save(train_data, f'{uglobals.TRAINING_DIR}/train_{self.layer}.pt')
        torch.save(val_data, f'{uglobals.TRAINING_DIR}/val_{self.layer}.pt')

    def prep_data(self):
        data = {} 
        # {sent_idx, [hidden_states, singular]} 
        # singular = 1

        for chunk_idx, chunk_name in enumerate(sorted(os.listdir(uglobals.COLORLESS_GREEN_HIDDEN_STATES_DIR), key=lambda x: int(x.split('_')[-1][:-3]))):
            chunk = torch.load(f'{uglobals.COLORLESS_GREEN_HIDDEN_STATES_DIR}/{chunk_name}')

            for sent_idx in sorted(chunk.keys()):
                hidden_states = chunk[sent_idx]
                hidden_state = hidden_states[self.layer][0, :, :] # (seq_len, hidden_size)

                # Get the correct subject number
                correct_number = self.metadata.iloc[sent_idx]['correct_number']
                if correct_number == 'sing':
                    sing = 1
                elif correct_number == 'plur':
                    sing = 0
                else:
                    raise ValueError(f'Invalid correct number: {correct_number}')

                # Keep only the part in the context (starting from the subject)
                prefix = self.metadata.iloc[sent_idx]['prefix']
                len_context = self.metadata.iloc[sent_idx]['len_context']
                discarded = ' '.join(prefix.split(' ')[: -len_context])
                discarded_len = len(self.tokenizer(discarded)['input_ids'])

                data[sent_idx] = {
                    'hidden_state': hidden_state[discarded_len: ],
                    'singular': sing
                }
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]
    
def hidden_states_collate(batch):
    hidden_state = torch.cat([b['hidden_state'].unsqueeze(0) for b in batch], dim=0)
    singular = torch.tensor([b['singular'] for b in batch])
    return {
        'hidden_state': hidden_state,
        'singular': singular
    }
    
def make_hidden_states_loader(path, layer, batch_size, shuffle=True):
    dataset = HiddenStatesDataset(layer, load_path=path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=hidden_states_collate)
    return loader