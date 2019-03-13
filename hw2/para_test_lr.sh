


if true ; then

    for opt in "adam" "ada";
    do

        for norm in "standard" "min_max" "mean" "none";
        do

            for batch_size in 10 50 100 500;
            do
                for l2_lambda in 0.1 0.01 0.001 0.0001;
                do

                    for eta in 10 1 0.1 0.01 0.001 0.0001 0.00001;
                    do

                        echo "optimizer "$opt" norm "$norm" batch_size "$batch_size" l2_lambda "$l2_lambda" eta "$eta

                        python3 logistic_regression.py \
                            --train_feature ./feature/X_train \
                            --train_answer ./feature/Y_train \
                            --validate 0.1 \
                            --random_seed 1234 \
                            --epoch 1000 \
                            --batch_size $batch_size \
                            --eta $eta \
                            --l2_lambda $l2_lambda \
                            --optimizer $opt \
                            --beta_m 0.9 \
                            --beta_v 0.999 \
                            --epsilon 1e-8 \
                            --norm $norm \
                            --early_stop true \
                            --out_log ./log/para_test_lg.log \
                            --debug true 2>&1 > /dev/null &
                            #--in_model \
                            #--out_model \
                            #--out_predict \

                    done

                    wait

                done
            done

        done

    done

fi
