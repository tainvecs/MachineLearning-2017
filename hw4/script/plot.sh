

# loss accuracy curves
if true ; then

    python3 ../src/plot-loss_acc_curves.py \
        --log_dir ../log \
        --output_dir ../plot/loss_acc_curves

fi

# plot model
if true ; then


    python3 ../src/plot-model.py \
        --model_dir ../model \
        --output_dir ../plot/model

fi


