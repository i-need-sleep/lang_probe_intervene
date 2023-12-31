import os
import argparse

import pandas as pd
import torch

import models.probes
import models.opt_intervention_patch as opt_intervention

import utils.globals as uglobals
import utils.data_utils as data_utils

def main(probe_type, probe_name, starting_layer, original_only, indices_dir):
    # Setup
    
    lr = 1e-3
    cutoff = 0.35 # The maximal distance to the set direnction

    print(f'Probe: {probe_name}, starting layer: {starting_layer}, lr: {lr}, cutoff: {cutoff}, original only: {original_only}, indices: {indices_dir}')

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load the probes
    probes = []
    for layer in range(25):
        if probe_type == 'linear':
            probe = models.probes.LinearProbe()
        elif probe_type == 'mlp':
            probe = models.probes.MLPProbe()
        else:
            raise ValueError(f'Unknown probe type: {probe_type}')
        checkpoint_dir = f'{uglobals.CHECKPOINTS_DIR}/{probe_name}/{probe_name}_layer{layer}'
        name = os.listdir(checkpoint_dir)[0]
        checkpoint_path = f'{checkpoint_dir}/{name}'
        checkpoint = torch.load(checkpoint_path)
        probe.load_state_dict(checkpoint['model_state_dict'])
        probe.to(device)
        probes.append(probe)

    
    # Load a patched OPT
    tokenizer, opt, intervention = opt_intervention.make_patched_opt(probes, starting_layer, lr, cutoff, uglobals.OPT_TOKENIZER_DIR, uglobals.OPT_MODEL_DIR, device)
    opt.to(device)

    # Data
    df = pd.read_csv(uglobals.COLORLESS_GREEN_PATH, sep='\t', header=0)
    val_dict = torch.load(indices_dir)
    val_indices = list(val_dict.keys())

    # Filter the indices
    filtered_indices = []
    for val_indice in val_indices:
        if original_only:
            if df.iloc[val_indice]['type'] == 'original':
                filtered_indices.append(val_indice)
        else:
            filtered_indices.append(val_indice)
    
    # Eval loop
    prob_diffs = []
    correct_ranks = []
    incorrect_ranks = []
    cases_skipped = 0

    for val_indice in filtered_indices:
        line_correct = df.iloc[val_indice]
        line_incorrect = df.iloc[val_indice + 1]

        prefix_incorrect = f'{line_incorrect["prefix"]} {line_incorrect["form"]}'
        prefix_correct = f'{line_correct["prefix"]} {line_correct["form"]}'

        # Intervene in the incorrect direction
        direction = 1 if line_incorrect['correct_number'] == 1 else 0

        # Find the shared prefix
        # We intervene at the point where the incorrect and correct sentences diverge
        incorrect_ids = tokenizer(prefix_incorrect, return_tensors='pt')['input_ids']
        correct_ids = tokenizer(prefix_correct, return_tensors='pt')['input_ids']

        prefix_len = 999
        for i in range(incorrect_ids.shape[1]):
            if i >= correct_ids.shape[1]:
                prefix_len = 999
                break
            if incorrect_ids[0, i] != correct_ids[0, i]:
                prefix_len = i
                break

        if prefix_len > max([len(prefix_incorrect), len(prefix_correct)]) or prefix_len == 999:
            cases_skipped += 1
            continue

        id_correct = correct_ids[0, prefix_len]
        id_incorrect = incorrect_ids[0, prefix_len]


        ids = incorrect_ids[:, : prefix_len].to(device)
        mask = tokenizer(prefix_incorrect, return_tensors='pt')['attention_mask'][:, : prefix_len].to(device)
        
        # Forward pass and intervention
        probs = forward_and_intervene(ids, mask, tokenizer, opt, intervention, direction)
        
        # Statistics
        prob_diff = probs[id_incorrect] - probs[id_correct]

        id_ranks = torch.argsort(probs, descending=True)
        correct_rank = torch.where(id_ranks == id_correct)[0][0]
        incorrect_rank = torch.where(id_ranks == id_incorrect)[0][0]

        prob_diffs.append(prob_diff)
        correct_ranks.append(correct_rank)
        incorrect_ranks.append(incorrect_rank)

    print(f'Cases skipped: {cases_skipped}')
    print(f'Remaining cases: {len(prob_diffs) - cases_skipped}')
    print(f'Avg prob diff: {sum(prob_diffs) / len(prob_diffs)}')
    print(f'Avg correct rank: {sum(correct_ranks) / len(correct_ranks)}')
    print(f'Avg incorrect rank: {sum(incorrect_ranks) / len(incorrect_ranks)}')
        
    return

def forward_and_intervene(ids, mask, tokenizer, opt, intervention, direction):
    intervention.set_intervention_direction(direction)
    probs = torch.nn.Softmax(dim=0)(opt(ids, mask)[0][0, -1, :]).cpu()

    return probs

if __name__ == '__main__':

    # cases = [f'n_attractor_{i}.pt' for i in range(4)] + ['len_context_less_than_7.pt', 'len_context_greatereq_than_7.pt']
    # for idx, case in enumerate(cases):
    #     cases[idx] = 'breakdown/' + case
    cases = ['training/train_0.pt', 'training/val_0.pt'] # Used to filter the test cases

    for case in cases:
        print(case)
        for original_only in [True]: # original_only = True # Only use non-synthesized texts
            for (probe_type, probe_name) in [
                ('linear', 'linear_3e-3'),
                ('mlp', 'mlp_3e-3')
                ]:
                for starting_layer in range(0, 26, 5): # 25 for the no intervention baseline
                    main(probe_type, probe_name, starting_layer, original_only, f'{uglobals.PROCESSED_DIR}/{case}')