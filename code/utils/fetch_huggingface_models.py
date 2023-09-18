from transformers import AutoTokenizer, AutoModel

import globals as uglobals

pretrained_dir = f'../{uglobals.PRETRAINED_DIR}'

download_name = r'facebook/opt-1.3b'
save_name = 'opt-1.3b'

tokenizer = AutoTokenizer.from_pretrained(download_name, force_download=True).save_pretrained(f'{pretrained_dir}/tokenizers/{save_name}')
model = AutoModel.from_pretrained(download_name, force_download=True).save_pretrained(f'{pretrained_dir}/models/{save_name}')