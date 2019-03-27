

# loss accuracy curves
if true ; then

    python3 ../src/plot-loss_acc_curves.py \
        --log_dir ../log/trained \
        --output_dir ../plot/loss_acc_curves

    python3 ../src/plot-loss_acc_curves.py \
        --log_dir ../log/sliced \
        --output_dir ../plot/loss_acc_curves

fi


# plot model
if true ; then


    python3 ../src/plot-model.py \
        --model_dir ../model/trained \
        --output_dir ../plot/model

fi


# activate filter
if true ; then

    python3 ../src/plot-conv_filter_visualization.py \
        --model ../model/best/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.hdf5 \
        --data ../data/train.csv \
        --output_dir ../plot/activate_filter

fi


# confusion_matrix
if true ; then

    python3 ../src/plot-confusion_matrix.py \
        --model ../model/best/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.hdf5 \
        --data ../data/train.csv \
        --output ../plot/confusion_matrix/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.confusion_matrix.png

fi


# saliency_map
if true ; then

    python3 ../src/plot-saliency_map.py \
        --model ../model/best/val_0.1_seed_1234_epoch_2000_batch_64_opt_adam_cnn_filter_112_elu_1.0_dnn_unit_896_elu_1.0_0722-0.7052.hdf5 \
        --data ../data/train.csv \
        --out_dir ../plot/saliency_map

fi
