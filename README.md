CUDA_VISIBLE_DEVICES=1 python train.py -p config/single_accent_config/preprocess_config.yaml -m config/single_accent_config/model_config.yaml -t config/single_accent_config/train_config.yaml --save_name american_english --restore_step 150000

CUDA_VISIBLE_DEVICES=3 python train.py -p config/single_accent_config_larger_encoder/preprocess_config.yaml -m config/single_accent_config_larger_encoder/model_config.yaml -t config/single_accent_config_larger_encoder/train_config.yaml --save_name american_english_large

CUDA_VISIBLE_DEVICES=0 python train.py -p config/single_accent_config_british/preprocess_config.yaml -m config/single_accent_config_british/model_config.yaml -t config/single_accent_config_british/train_config.yaml --save_name british_english 
