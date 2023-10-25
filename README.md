# lang_probe_intervene

## LM
* Use [code/utils/fetch_huggingface_models.py](https://github.com/i-need-sleep/lang_probe_intervene/blob/main/code/utils/fetch_huggingface_models.py) to download the model/tokenizer checkpoints from Huggingface.

## Data
* Download annotated sentences from [here](https://drive.google.com/drive/folders/1VHWIOWqr4TU6uzn-SoFLUDNFpF-nbjYd?usp=sharing) and put it under data/colorless_green
* Compute and store the OPT hidden states at each layer with [code/get_opt_hidden_states.py](https://github.com/i-need-sleep/lang_probe_intervene/blob/main/code/get_opt_hidden_states.py)

## Training
* With [code/train_probes.py](https://github.com/i-need-sleep/lang_probe_intervene/blob/main/code/train_probes.py).
* See the probe configurations under [code/models/probes.py](https://github.com/i-need-sleep/lang_probe_intervene/blob/main/code/models/probes.py)

## Intervention
* With [code/intervene_opt.py](https://github.com/i-need-sleep/lang_probe_intervene/blob/main/code/intervene_opt.py)
* See [code/models/opt_intervention_patch.py](https://github.com/i-need-sleep/lang_probe_intervene/blob/main/code/models/opt_intervention_patch.py) for a patched OPT forward functions applying intervention.