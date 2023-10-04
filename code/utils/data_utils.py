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
        for sent_idx, line in data.items():
            hidden_states = line['hidden_state']
            singular = line['singular']
            for i in range(hidden_states.shape[0]):
                out.append({
                    'hidden_state': hidden_states[i, :],
                    'singular': singular,
                    'send_idx': sent_idx,
                    'token_idx': i,
                    'context_token_len': hidden_states.shape[0],
                    'token': line['tokens'][i]
                })
        return out

    def make_splits(self, val_ratio=0.1):
        data, original_ids = self.prep_data() # {sent_idx, [hidden_states]}

        # Split into train, val
        # indices = list(data.keys())
        # random.shuffle(indices)
        # val_size = int(val_ratio * len(indices))
        # val_indices = indices[: val_size]
        
        # train_data = {}
        # val_data = {}
        # for sent_idx in data.keys():
        #     if sent_idx in val_indices:
        #         val_data[sent_idx] = data[sent_idx]
        #     else:
        #         train_data[sent_idx] = data[sent_idx]

        # Make train/val splits by original ids
        indices = list(range(len(original_ids)))
        random.shuffle(indices)
        val_size = int(val_ratio * len(indices))
        val_indices = indices[: val_size]
        val_original_ids = [original_ids[i] for i in val_indices]
        
        # Such that all layers will follow the same splits 
        val_original_ids = ['R__NOUN_NOUN_PUNCT_CCONJ_NOUN_NOUN_3', 'R__NOUN_PROPN_PROPN_ADP_VERB_0', 'R__NOUN_NOUN_PUNCT_CCONJ_NOUN_NOUN_4', 'R__VERB_NOUN_CCONJ_VERB_4', 'L__AUX_DET_ADJ_ADJ_NOUN_10', 'L__DET_ADJ_NOUN_1', 'L__DET_ADJ_NOUN_0', 'R__NOUN_PROPN_PROPN_ADP_VERB_12', 'R__VERB_NOUN_CCONJ_VERB_2', 'R__NOUN_ADJ_PUNCT_NOUN_0', 'R__NOUN_PROPN_PROPN_PRON_VERB_1', 'R__VERB_SCONJ_NOUN_VERB_0', 'L__AUX_DET_ADJ_ADJ_NOUN_8', 'R__VERB_VERB_CCONJ_PRON_VERB_1', 'R__NOUN_ADJ_PUNCT_NOUN_1', 'L__AUX_DET_NOUN_NOUN_2', 'R__VERB_PRON_DET_PROPN_NOUN_7', 'L__AUX_DET_ADJ_ADJ_NOUN_1', 'R__NOUN_PROPN_PROPN_ADP_VERB_8', 'R__NOUN_PROPN_PROPN_ADP_VERB_23', 'R__VERB_NOUN_ADP_PRON_0']

        train_data = {}
        val_data = {}
        for sent_idx, datum in data.items():
            if datum['original_id'] in val_original_ids:
                val_data[sent_idx] = datum
            else:
                train_data[sent_idx] = datum

        torch.save(train_data, f'{uglobals.TRAINING_DIR}/train_{self.layer}.pt')
        torch.save(val_data, f'{uglobals.TRAINING_DIR}/val_{self.layer}.pt')

    def prep_data(self):
        data = {} 
        # {sent_idx, [hidden_states, singular]} 
        # singular = 1

        # Keep track of the pattern
        original_ids = []
        
        for chunk_idx, chunk_name in enumerate(sorted(os.listdir(uglobals.COLORLESS_GREEN_HIDDEN_STATES_DIR), key=lambda x: int(x.split('_')[-1][:-3]))):
            chunk = torch.load(f'{uglobals.COLORLESS_GREEN_HIDDEN_STATES_DIR}/{chunk_name}')

            for sent_idx in sorted(chunk.keys()):
                # Correct/incorrect duplicates for the same sentence
                if sent_idx % 2 == 1:
                    continue
                
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

                original_id = self.metadata.iloc[sent_idx]['pattern'] + '_' + str(self.metadata.iloc[sent_idx]['constr_id'])
                if original_id not in original_ids:
                    original_ids.append(original_id)
                patterns_idx = original_ids.index(original_id)

                data[sent_idx] = {
                    'hidden_state': hidden_state[discarded_len: ],
                    'singular': sing,
                    'tokens': self.tokenizer(prefix)['input_ids'][discarded_len: ],
                    'original_id': original_id,
                }
        return data, original_ids
    
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

def get_indices_by_n_attractors():
    df = pd.read_csv(uglobals.COLORLESS_GREEN_PATH, sep='\t', header=0)
    max_n_attractor = df['n_attr'].max()

    for n_attractor in range(max_n_attractor + 1):
        indices = df[df['n_attr'] == n_attractor].index

        out = {}
        for index in indices:
            if index % 2 == 0:
                out[index] = True
        
        torch.save(out, f'{uglobals.PROCESSED_DIR}/breakdown/n_attractor_{n_attractor}.pt')
    return

def get_indices_by_context_length(cutoff=7):
    df = pd.read_csv(uglobals.COLORLESS_GREEN_PATH, sep='\t', header=0)

    indices = df[df['len_context'] < cutoff].index

    out = {}
    for index in indices:
        if index % 2 == 0:
            out[index] = True
        
    torch.save(out, f'{uglobals.PROCESSED_DIR}/breakdown/len_context_less_than_{cutoff}.pt')

    indices = df[df['len_context'] >= cutoff].index

    out = {}
    for index in indices:
        if index % 2 == 0:
            out[index] = True
        
    torch.save(out, f'{uglobals.PROCESSED_DIR}/breakdown/len_context_greatereq_than_{cutoff}.pt')
    return