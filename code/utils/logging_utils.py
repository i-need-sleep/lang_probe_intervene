import os

from .globals import *

def parse_probing_logs():
    for probe_type in ['linear', 'mlp']:

        out = {} # {layer: [best_acc, lr]}
        for i in range(25):
            out[i] = [0, None]

        for lr in ['-3', '-4', '-5', '-6', '-7']:
            print(f'===== Probe type: {probe_type}, Lr: {lr} =====')
            for log_name in os.listdir(PROBING_LOGS_DIR):
                if probe_type in log_name and 'out' in log_name and f'3e{lr}' in log_name:
                    log_out = parse_log(f'{PROBING_LOGS_DIR}/{log_name}')
                    for i, acc in enumerate(log_out):
                        if acc > out[i][0]:
                            out[i] = [acc, f'we{lr}']
        for key, val in out.items():
            val[0] = "%.3f" % val[0]
            print(f'Layer {key}: {val}')
        
    return

def parse_log(path):
    out = []
    # [best_acc_layer_0, best_acc_layer_1, ...]
    
    with open(path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if '-----Probing layer' in line:
            out.append(0)
        if 'Dev accuracy: ' in line:
            dev_acc = float(line.split(' ')[-1])
            if dev_acc > out[-1]:
                out[-1] = dev_acc
    
    return out