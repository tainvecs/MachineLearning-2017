

python3 ../src/semi-supervised.py \
    --w2v ../w2v/cw_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all \
    --data ../data/semi/x_train.non_labeled \
    --val_thres 0.83 \
    --cfd_upper_bound 1 \
    --cfd_lower_bound 0.95 \
    --model_dir ../model \
    --out_x ../data/semi/x_train.semi.val_0.1_seed_1234_val_lgrthn_0.83_cfd_0.95 \
    --out_y ../data/semi/y_train.semi.val_0.1_seed_1234_val_lgrthn_0.83_cfd_0.95.txt \
    --out_x_prob ../data/semi/x_train_prob.semi.val_0.1_seed_1234_val_lgrthn_0.83


python3 ../src/semi-supervised.py \
    --w2v ../w2v/cw_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all \
    --data ../data/semi/x_train.non_labeled \
    --cfd_upper_bound 0.7 \
    --cfd_lower_bound 0.3 \
    --in_x_prob ../data/semi/x_train_prob.semi.val_0.1_seed_1234_val_lgrthn_0.83.npy \
    --out_x ../data/semi/x_train.semi.val_0.1_seed_1234_val_lgrthn_0.83_cfd_0.3_0.7 \
    --out_y ../data/semi/y_train.semi.val_0.1_seed_1234_val_lgrthn_0.83_cfd_0.3_0.7.txt

