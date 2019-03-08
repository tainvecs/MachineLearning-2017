


if true ; then

    for norm in "standard" "min_max" "mean" "none";
    do
    
        echo "norm "$norm
        
        python3 probabilistic_generative_model.py \
            --train_x ./feature/X_train \
            --train_y ./feature/Y_train \
            --validate 0.1 \
            --random_seed 1234 \
            --norm $norm \
            --test_x ./feature/X_test \
            --debug false

    done

fi
