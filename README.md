# Accent Resynthesis

## Setup
Note that this repo has been tested on Python >3.11.
```
conda create -n <env_name> python=3.13
pip install -r requirements.txt
```
## Training
See SPARC paper for details on how to generate SPARC features.
See UniG2P repo for generating Unilex phonemes.

Once both of these are created, run training with the appropriate model, preprocess, and training parameters.

Example command:
```
python train.py -p config/single_accent_config/preprocess_config.yaml -m config/single_accent_config/model_config.yaml -t config/single_accent_config/train_config.yaml --save_name american_english --restore_step 150000
```
## Inference
See `testing.ipynb`.

python train.py -p config/2_accent_config/preprocess_config.yaml -m config/2_accent_config/model_config.yaml -t config/2_accent_config/train_config.yaml --save_name sepconv_aligner

Audio samples can be found in the `audio_samples` directory.