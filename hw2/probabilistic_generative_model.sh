
python3 probabilistic_generative_model.py \
    --train_x ./feature/X_train \
    --train_y ./feature/Y_train \
    --validate 0.1 \
    --random_seed 1234 \
    --norm standard \
    --test_x ./feature/X_test \
    --out_predict ./tmp.out.csv \
    --debug true
