

python3 logistic_regression.py \
    --train_feature ./feature/X_train \
    --train_answer ./feature/Y_train \
    --test_feature ./feature/X_test \
    --validate 0.1 \
    --random_seed 1234 \
    --epoch 1000 \
    --batch_size 50 \
    --eta 0.1 \
    --l2_lambda 0.0001 \
    --optimizer 'adam' \
    --beta_m 0.9 \
    --beta_v 0.999 \
    --epsilon 1e-8 \
    --norm 'standard' \
    --early_stop true \
    --debug true
    # --test_answer \
    #--in_model \
    # --out_log \
    #--out_model \
    #--out_predict \
