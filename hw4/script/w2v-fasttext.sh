

make -C ../site-package/fastText clean fasttext


############################## x_train.all.raw.txt #############################
# fast text skip-gram embedding, subword
../site-package/fastText/fasttext skipgram \
    -input ../data/preprocessed/x_train.all.raw.txt \
    -bucket 300000 \
    -dim 256 \
    -loss ns \
    -neg 5 \
    -ws 5 \
    -lr 0.1 \
    -epoch 30 \
    -wordNgrams 3 \
    -minn 3 \
    -maxn 6 \
    -thread 16 \
    -output ../model/sg_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all.raw

# fast text cbow embedding, subword
../site-package/fastText/fasttext cbow \
    -input ../data/preprocessed/x_train.all.raw.txt \
    -bucket 300000 \
    -dim 256 \
    -loss ns \
    -neg 5 \
    -ws 5 \
    -lr 0.1 \
    -epoch 30 \
    -wordNgrams 3 \
    -minn 3 \
    -maxn 6 \
    -thread 16 \
    -output ../model/cw_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all.raw


############################# x_train.all.proc.txt #############################
# fast text skip-gram embedding, subword
../site-package/fastText/fasttext skipgram \
    -input ../data/preprocessed/x_train.all.proc.txt \
    -bucket 300000 \
    -dim 256 \
    -loss ns \
    -neg 5 \
    -ws 5 \
    -lr 0.1 \
    -epoch 30 \
    -wordNgrams 3 \
    -minn 3 \
    -maxn 6 \
    -thread 16 \
    -output ../model/sg_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all.proc

# fast text cbow embedding, subword
../site-package/fastText/fasttext cbow \
    -input ../data/preprocessed/x_train.all.proc.txt \
    -bucket 300000 \
    -dim 256 \
    -loss ns \
    -neg 5 \
    -ws 5 \
    -lr 0.1 \
    -epoch 30 \
    -wordNgrams 3 \
    -minn 3 \
    -maxn 6 \
    -thread 16 \
    -output ../model/cw_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all.proc


########################### x_train.all.freq_proc.txt ##########################
# fast text skip-gram embedding, subword
../site-package/fastText/fasttext skipgram \
    -input ../data/preprocessed/x_train.all.freq_proc.txt \
    -bucket 300000 \
    -dim 256 \
    -loss ns \
    -neg 5 \
    -ws 5 \
    -lr 0.1 \
    -epoch 30 \
    -wordNgrams 3 \
    -minn 3 \
    -maxn 6 \
    -thread 16 \
    -output ../model/sg_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all.freq_proc

# fast text cbow embedding, subword
../site-package/fastText/fasttext cbow \
    -input ../data/preprocessed/x_train.all.freq_proc.txt \
    -bucket 300000 \
    -dim 256 \
    -loss ns \
    -neg 5 \
    -ws 5 \
    -lr 0.1 \
    -epoch 30 \
    -wordNgrams 3 \
    -minn 3 \
    -maxn 6 \
    -thread 16 \
    -output ../model/cw_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all.freq_proc


######################## x_train.all.stem_frq_pro.txt ########################
# fast text skip-gram embedding, subword
../site-package/fastText/fasttext skipgram \
    -input ../data/preprocessed/x_train.all.stem_frq_pro.txt \
    -bucket 300000 \
    -dim 256 \
    -loss ns \
    -neg 5 \
    -ws 5 \
    -lr 0.1 \
    -epoch 30 \
    -wordNgrams 3 \
    -minn 3 \
    -maxn 6 \
    -thread 16 \
    -output ../model/sg_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all.stem_frq_pro

# fast text cbow embedding, subword
../site-package/fastText/fasttext cbow \
    -input ../data/preprocessed/x_train.all.stem_frq_pro.txt \
    -bucket 300000 \
    -dim 256 \
    -loss ns \
    -neg 5 \
    -ws 5 \
    -lr 0.1 \
    -epoch 30 \
    -wordNgrams 3 \
    -minn 3 \
    -maxn 6 \
    -thread 16 \
    -output ../model/cw_256_neg_5_ws_5_lr_0.1_epoch_30_ngram_3_char_3_6_thread_16.x_train.all.stem_frq_pro
