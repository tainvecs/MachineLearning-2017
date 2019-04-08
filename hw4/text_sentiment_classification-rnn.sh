
python3 text_sentiment_classification-rnn.py \
    --x_train ./data/preprocessed/x_train.labeled.proc.txt \
    --y_train ./data/preprocessed/y_train.labeled.txt \
    --validate 0.1 \
    --random_seed 1234 \
    --w2v ./w2v/cw_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all.proc \
    --model lstm \
    --epoch 20 \
    --batch_size 64 \
    --optimizer adam \
    --rnn_activation tanh \
    --dnn_activation swish \
    --dnn_norm true
    # --out_log  \
    # --out_model

