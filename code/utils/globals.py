# Data
DATA_DIR = '../data'
PROCESSED_DIR = f'{DATA_DIR}/processed'
TRAINING_DIR = f'{PROCESSED_DIR}/training'

COLORLESS_GREEN_PATH = f'{DATA_DIR}/colorless_green/generated.tab'
COLORLESS_GREEN_HIDDEN_STATES_DIR = f'{PROCESSED_DIR}/colorless_green'

# Pretrained checkpoints
PRETRAINED_DIR = '../pretrained'

OPT_MODEL_DIR = f'{PRETRAINED_DIR}/models/opt-1.3b'
OPT_TOKENIZER_DIR = f'{PRETRAINED_DIR}/tokenizers/opt-1.3b'

# Results
RESULTS_DIR = '../results'
CHECKPOINTS_DIR = f'{RESULTS_DIR}/checkpoints'
OUTPUTS_DIR = f'{RESULTS_DIR}/outputs'

# Logs
LOGS_DIR = '../sh/logs'
PROBING_LOGS_DIR = f'{LOGS_DIR}/probing'