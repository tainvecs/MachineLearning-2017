

# loss accuracy curves
if true ; then
    
    for cfd in cfd_0.3_0.7  cfd_0.5_0.7  cfd_0.7  cfd_0.95 ;
    do 
        python3 ../src/plot-loss_acc_curves.py \
            --log_dir ../log-semi/$cfd \
            --output_dir ../plot-semi/loss_acc_curves/$cfd
    done

fi

