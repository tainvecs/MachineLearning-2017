
# val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0.log
python3 image_sentiment_classification-cnn.py \
    --train ./data/train.csv \
    --validate 0.1 \
    --random_seed 1234 \
    --epoch 2000 \
    --batch_size 64 \
    --optimizer adam \
    --filters 112 \
    --cnn_activation ELU \
    --cnn_activation_alpha 1.0 \
    --units 896 \
    --dnn_activation ELU \
    --dnn_activation_alpha 1.0
    # --out_log \
    # --out_model \
