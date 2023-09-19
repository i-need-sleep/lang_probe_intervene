from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, OPTForCausalLM

import utils.globals as uglobals

def main(debug=False):

    if torch.cuda.is_available() and not debug:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the tokeniser and the model
    tokenizer = AutoTokenizer.from_pretrained(uglobals.OPT_TOKENIZER_DIR)
    model = OPTForCausalLM.from_pretrained(uglobals.OPT_MODEL_DIR).to(device)

    # Load data
    data = pd.read_csv(f'{uglobals.COLORLESS_GREEN_PATH}', sep='\t', header=0)

    # Get the hidden states
    out_list = []
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(len(data))):
            if debug and i > 2:
                break
            prefix_text = data.iloc[i]['prefix']
            tokenized = tokenizer(prefix_text, return_tensors='pt').to(device)
            hidden_states = model(tokenized['input_ids'], tokenized['attention_mask'], output_hidden_states=True).hidden_states.cpu()
            # One for the input embeddings + one for each layer
            out_list.append(hidden_states)

    torch.save(out_list, f'{uglobals.PROCESSED_DIR}/colorless_opt_hidden_states.pt')

if __name__ == '__main__':
    main()