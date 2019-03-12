

# adam
if true ; then

    for batch_size in 10 50 100 500 1000;
    do
        for l2_lambda in 0 1 0.1 0.01 0.001 0.0001;
        do

            echo "#-------------------------adam-------------------------#"

            for eta in 10 1 0.1 0.01 0.001 0.0001;
            do

                echo "batch_size "$batch_size" l2_lambda "$l2_lambda" eta "$eta

                python3 linear_regression.py \
                    --train ./data/train.csv \
                    --validate 500 \
                    --random_seed 1234 \
                    --epoch 10000 \
                    --batch_size $batch_size \
                    --eta $eta \
                    --l2_lambda $l2_lambda \
                    --optimizer adam \
                    --beta_m 0.9 \
                    --beta_v 0.999 \
                    --epsilon 1e-8 \
                    --early_stop true \
                    --debug true \
                    --out_log ./log/adam.log 2>&1 > /dev/null &
                    # --test ./data/test.csv \
                    # --out_model
                    # --out_predict

            done

            wait

        done
    done

fi



# ada
if false ; then

    for batch_size in 10 50 100 500 1000;
    do
        for l2_lambda in 0 1 0.1 0.01 0.001 0.0001;
        do

            echo "#-------------------------ada-------------------------#"

            for eta in 10 1 0.1 0.01 0.001 0.0001;
            do

                echo "batch_size "$batch_size" l2_lambda "$l2_lambda" eta "$eta

                python3 linear_regression.py \
                    --train ./data/train.csv \
                    --validate 500 \
                    --random_seed 1234 \
                    --epoch 10000 \
                    --batch_size $batch_size \
                    --eta $eta \
                    --l2_lambda $l2_lambda \
                    --optimizer ada \
                    --epsilon 1e-8 \
                    --early_stop true \
                    --debug true \
                    --out_log ./log/ada.log 2>&1 > /dev/null &
                    # --test ./data/test.csv \
                    # --out_model
                    # --out_predict

            done

            wait

        done
    done

fi
