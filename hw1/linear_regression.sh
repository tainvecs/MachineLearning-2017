

python3 linear_regression.py \
    --train ./data/train.csv \
    --test ./data/test.csv \
    --validate 500 \
    --random_seed 1234 \
    --epoch 10000 \
    --batch_size 50 \
    --eta 0.01 \
    --l2_lambda 0.001 \
    --optimizer adam \
    --epsilon 1e-8 \
    --early_stop true \
    --debug true
    # --out_log
    # --out_model
    # --out_predict
