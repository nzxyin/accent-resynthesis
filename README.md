# Accent Resynthesis

## Setup
```
conda create -n <env_name> python=3.12
pip install -r requirements.txt
```
## Training
Requires MFA extracted durations, see docs [here](https://montreal-forced-aligner.readthedocs.io/en/latest/).

Example command:
```
python train.py -p config/single_accent_config/preprocess_config.yaml -m config/single_accent_config/model_config.yaml -t config/single_accent_config/train_config.yaml --save_name american_english --restore_step 150000
```
## Inference
See `testing.ipynb`.
