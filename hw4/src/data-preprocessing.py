

import argparse
from gensim.parsing.porter import PorterStemmer

import re
import os
from collections import defaultdict

from util.pruning import PruneDict, PrunePunctuation, PruneNumber
from util.pruning import PruneDuplicateCharacter, PruneContraction, PruneCategory, PruneRareWord


def ParseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_labeled', help='')
    parser.add_argument('--train_non_labeled', help='')
    parser.add_argument('--test', help='')
    parser.add_argument('--out_dir', help='')

    args = parser.parse_args()


    print ("################## Arguments ##################")

    args_dict = vars(args)
    for key in args_dict:
        print ( "\t{}: {}".format(key, args_dict[key]) )

    print ("###############################################")

    return args


def CountFrequency(text_list):

    senlen_freq, word_freq, char_freq = defaultdict(int), defaultdict(int), defaultdict(int)
    for text in text_list:

        words = text.split()
        senlen_freq[len(words)] += 1

        for word in words:

            word_freq[word] += 1
            for char in word:
                char_freq[char] += 1

    return senlen_freq, word_freq, char_freq


def LoadLabeledTrain(path):

    label_list, text_list = [], []

    with open(path, 'r') as in_file:
        for line in in_file:
            label, text = line.split('+++$+++')
            label_list.append(label.strip())
            text_list.append(text.strip())

    return label_list, text_list

def PreprocessTwitterText(text, word_freq, word_freq_thres=80):

    words = text.split()
    sen, prune_dict = [], PruneDict()

    for i in range(len(words)):

        word = words[i]

        if word in prune_dict:
            word = prune_dict[word]

        word = PrunePunctuation(word)
        word = PruneNumber(word)
        word = PruneDuplicateCharacter(word, word_freq, word_freq_thres)
        word = PruneCategory(word)

        if (i>=2) and (words[i-1] == "\'"):
            pre_word, post_word = PruneContraction(words[i-2], word)
            sen[-2], sen[-1], word = pre_word, post_word, None

        if word in prune_dict:
            word = prune_dict[word]

        if not (word is None):
            sen.append(word)

    return ' '.join(sen)


if __name__ == '__main__':


    args = ParseArgs()


    # labeled training data
    print ("\nPre-processing Labeled Training Data...")

    # raw
    label_lb_list, text_lb_list = LoadLabeledTrain(args.train_labeled)
    _, word_freq, _ = CountFrequency(text_lb_list)
    with open(os.path.join(args.out_dir, "x_train.labeled.raw.txt"), 'w') as out_file:
        out_file.write('\n'.join(text_lb_list))
    with open(os.path.join(args.out_dir, "y_train.labeled.txt"), 'w') as out_file:
        out_file.write('\n'.join(label_lb_list))

    # pre-process
    proc_text_lb_list = [ PreprocessTwitterText(text, word_freq, word_freq_thres=80) for text in text_lb_list ]
    _, proc_word_freq, _ = CountFrequency(proc_text_lb_list)
    with open(os.path.join(args.out_dir, "x_train.labeled.proc.txt"), 'w') as out_file:
        out_file.write('\n'.join(proc_text_lb_list))

    # filter rare word
    freq_proc_text_lb_list = [ PruneRareWord(text, proc_word_freq, freq_thres=5) for text in proc_text_lb_list ]
    _, freq_proc_word_freq, _ = CountFrequency(freq_proc_text_lb_list)
    with open(os.path.join(args.out_dir, "x_train.labeled.freq_proc.txt"), 'w') as out_file:
        out_file.write('\n'.join(freq_proc_text_lb_list))

    # stem
    p = PorterStemmer()
    stem_frq_pro_text_lb_list = p.stem_documents(freq_proc_text_lb_list)
    _, stem_frq_pro_word_freq, _ = CountFrequency(stem_frq_pro_text_lb_list)

    with open(os.path.join(args.out_dir, "x_train.labeled.stem_frq_pro.txt"), 'w') as out_file:
        out_file.write('\n'.join(stem_frq_pro_text_lb_list))

    print ("Labeled Training Data: \n\traw: {} word tokens\n\tprocessed: {} word tokens\n\tfrequent: {} word tokens\n\tstemmed: {} word tokens".format(len(word_freq), len(proc_word_freq), len(freq_proc_word_freq), len(stem_frq_pro_word_freq)))


    # non-labeled training data
    print ("\nPre-processing Non-Labeled Training Data...")

    # raw
    with open(args.train_non_labeled, 'r') as in_file:
        text_nlb_list = [ line.strip() for line in in_file ]
    _, word_freq, _ = CountFrequency(text_nlb_list)
    with open(os.path.join(args.out_dir, "x_train.all.raw.txt"), 'w') as out_file:
        out_file.write('\n'.join(text_lb_list + text_nlb_list))

    # pre-process
    proc_text_nlb_list = [ PreprocessTwitterText(text, word_freq, word_freq_thres=160) for text in text_nlb_list ]
    _, proc_word_freq, _ = CountFrequency(proc_text_nlb_list)
    with open(os.path.join(args.out_dir, "x_train.all.proc.txt"), 'w') as out_file:
        out_file.write('\n'.join(proc_text_lb_list + proc_text_nlb_list))

    # filter rare word
    freq_proc_text_nlb_list = [ PruneRareWord(text, proc_word_freq, freq_thres=5) for text in proc_text_nlb_list ]
    _, freq_proc_word_freq, _ = CountFrequency(freq_proc_text_nlb_list)
    with open(os.path.join(args.out_dir, "x_train.all.freq_proc.txt"), 'w') as out_file:
        out_file.write('\n'.join(freq_proc_text_lb_list + freq_proc_text_nlb_list))

    # stem
    p = PorterStemmer()
    stem_frq_pro_text_nlb_list = p.stem_documents(freq_proc_text_nlb_list)
    _, stem_frq_pro_word_freq, _ = CountFrequency(stem_frq_pro_text_nlb_list)
    with open(os.path.join(args.out_dir, "x_train.all.stem_frq_pro.txt"), 'w') as out_file:
        out_file.write('\n'.join(stem_frq_pro_text_lb_list + stem_frq_pro_text_nlb_list))

    print ("Non-Labeled Training Data: \n\traw: {} word tokens\n\tprocessed: {} word tokens\n\tfrequent: {} word tokens\n\tstemmed: {} word tokens".format(len(word_freq), len(proc_word_freq), len(freq_proc_word_freq), len(stem_frq_pro_word_freq)))


    # test data
    print ("\nProcessing Text Training Data...")
    with open(args.test, 'r') as in_file:
        in_file.readline()
        test_list = [ line.split(',', 1)[1].strip() for line in in_file ]
    with open(os.path.join(args.out_dir, "x_test.raw.txt"), 'w') as out_file:
        out_file.write('\n'.join(test_list))

    # pre-process
    proc_test_list = [ PreprocessTwitterText(text, word_freq, word_freq_thres=80) for text in test_list ]
    with open(os.path.join(args.out_dir, "x_test.proc.txt"), 'w') as out_file:
        out_file.write('\n'.join(proc_test_list))

    # filter rare word
    freq_proc_test_list = [ PruneRareWord(text, proc_word_freq, freq_thres=5) for text in proc_test_list ]
    with open(os.path.join(args.out_dir, "x_test.freq_proc.txt"), 'w') as out_file:
        out_file.write('\n'.join(freq_proc_test_list))

    # stem
    p = PorterStemmer()
    stem_frq_pro_test_list = p.stem_documents(freq_proc_test_list)
    with open(os.path.join(args.out_dir, "x_test.stem_frq_pro.txt"), 'w') as out_file:
        out_file.write('\n'.join(stem_frq_pro_test_list))

    print ("Done~")
