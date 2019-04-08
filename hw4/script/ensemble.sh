

python3 ../src/ensemble.py \
    --w2v ../w2v/cw_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all \
    --data ../data/preprocessed/x_train.labeled \
    --answer ../data/preprocessed/y_train.labeled.txt \
    --validate 0.1 \
    --seed 1234 \
    --val_thres 0.83 \
    --model_dir ../model \
    --output ../output/ensemble.val_0.1_seed_1234_val_lgrthn_0.83.csv
