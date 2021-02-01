#!/usr/bin/env python3

import argparse

import torch
import torch.nn.functional as F
import math
import numpy as np

from model.transformer import TransformerModel, TransformerHead
from preprocessing import get_batch, Corpus, batchify, batchify_docs, get_batch_docs, file_to_df
import torch.optim as optim
from eval import eval
import os
import gc
import pandas as pd
from nltk.stem.porter import *
import sentencepiece as spm
from sklearn import model_selection
import sys


def run_model(args):
    if not os.path.exists(args.trained_classification_models_dir):
        os.makedirs(args.trained_classification_models_dir)

    if not os.path.exists(args.trained_language_models_dir):
        os.makedirs(args.trained_language_models_dir)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    if not os.path.exists(args.dict_folder):
        os.makedirs(args.dict_folder)

    if not os.path.exists(os.path.join(args.result_path, 'predictions')):
        os.makedirs(os.path.join(args.result_path, 'predictions'))

    dict_files = os.listdir(args.dict_folder)
    dict_exists = False

    for dict in dict_files:
        if args.config_id in dict:
            dict_exists = True
            dict_path = os.path.join(args.dict_folder, dict)
            if args.transfer_learning:
                print('Using existing dictionary: ', dict)
            break
    if not args.transfer_learning:
        if args.rnn:
            drnn = 'rnn'
        else:
            drnn = 'nornn'
        if args.crf:
            dcrf = 'crf'
        else:
            dcrf = 'nocrf'
        if args.adaptive:
            dadaptive = 'adaptive'
        else:
            dadaptive = 'noadaptive'
        if args.masked_lm:
            dmasked_lm = 'maskedlm'
        else:
            dmasked_lm = 'lm'
        if args.bpe:
            dbpe = 'bpe'
        else:
            dbpe = 'nobpe'
        if args.POS_tags:
            dpos = 'pos'
        else:
            dpos = 'nopos'
        if dict_exists:
            print()
            print('---------------------------------------------------------')
            print('Dictionary for that config already exists, exiting!!!!!!!')
            print('---Remove the dictionary file or change the config id----')
            print('---------------------------------------------------------')
            print()
            sys.exit()
        else:
            if not args.classification:
                dict_path = os.path.join(args.dict_folder, args.config_id + '_' + dadaptive + '_' + dmasked_lm + '_' + dbpe + '_' + dpos + '_' + drnn + '_' + dcrf + '.ptb')
            else:
                dict_path = os.path.join(args.dict_folder, args.config_id + '_nolm_' + dbpe + '_' + dpos + '_' + drnn + '_' + dcrf + '.ptb')

    args.dict_path = dict_path

    if args.bpe:
        assert os.path.exists(args.bpe_model_path)
        sp = spm.SentencePieceProcessor()
        sp.Load(args.bpe_model_path)
    else:
        sp = None

    if args.crf:
        assert not args.rnn
    if args.rnn:
        assert not args.crf

    if args.classification:
        assert args.datasets is not None
        assert args.trained_classification_models_dir != args.trained_language_models_dir
        assert not args.adaptive
        if args.transfer_learning:
            l_models = os.listdir(args.trained_language_models_dir)
            for l_model in l_models:
                if args.config_id in l_model:
                    args.language_model_path = os.path.join(args.trained_language_models_dir, l_model)
            print('Classification, using language model: ', args.language_model_path)
            print()

    if not args.transfer_learning:
        if not args.classification:
            assert os.path.exists(os.path.join(args.data_path, args.lm_corpus_file))
        assert not os.path.exists(args.dict_path)

    print(args)

    stemmer = PorterStemmer()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not args.classification:
        df_data = file_to_df(os.path.join(args.data_path, args.lm_corpus_file), classification=False)
        df_data = df_data.sample(frac=1, random_state=args.seed)
        val_idx = int(0.8 * df_data.shape[0])
        test_idx = int(0.9 * df_data.shape[0])
        df_train = df_data[:val_idx]
        df_valid = df_data[val_idx:test_idx]
        df_test = df_data[test_idx:]

        print('------------------------------------------------------------------------------------------------------')
        print('Training language model on all data')
        print("Train size: ", df_train.shape, "Valid size: ", df_valid.shape, "Test size: ", df_test.shape)
        print('------------------------------------------------------------------------------------------------------')
        print()
        train_test(df_train, df_valid, df_test, args, stemmer, sp)

    else:
        result_file = open(os.path.join(args.result_path, 'classification_results.txt'), 'a', encoding='utf8')
        result_file.write("Classification results for config " + args.config_id + ":\n\n")
        result_file.write("Parameters:\n")
        result_file.write(str(args) + '\n------------------------------------------------\n')

        for folder in args.datasets.split(';'):

            print('------------------------------------------------------------------------------------------------------')
            print('Training on: ', folder)
            print('------------------------------------------------------------------------------------------------------')

            if folder in args.cv_eval.split(';'):
                #cross validation
                kf = model_selection.KFold(n_splits=10)
                df_data = file_to_df(os.path.join(args.data_path, folder, folder + '_test.json'), classification=True)
                df_data = df_data.sample(frac=1, random_state=args.seed)
                print()
                print('Cross validation')

                fold_counter = 0

                total_pred = []
                total_true = []

                for train_index, test_index in kf.split(df_data):
                    fold_counter += 1
                    df_train, df_test = df_data.iloc[train_index], df_data.iloc[test_index]
                    sep_idx = int(df_train.shape[0] / 10)
                    df_valid = df_train[:sep_idx]
                    df_train = df_train[sep_idx:]

                    print("Train fold ", fold_counter, "fold size: ", df_train.shape, "Valid fold size: ", df_valid.shape, "Test fold  size: ", df_test.shape)
                    print()

                    fold_pred, fold_true, num_parameters = train_test(df_train, df_valid, df_test, args, stemmer, sp, folder)
                    total_pred.extend(fold_pred)
                    total_true.extend(fold_true)
                print()
                print('--------------------------------------------------------------------')
                print('Final CV results:')
                print()

            else:
                df_train = file_to_df(os.path.join(args.data_path, folder, folder + '_valid.json'), classification=True)
                df_train = df_train.sample(frac=1, random_state=args.seed)
                val_idx = int(0.8 * df_train.shape[0])
                df_valid = df_train[val_idx:]
                df_train = df_train[:val_idx]
                df_test = file_to_df(os.path.join(args.data_path, folder, folder + '_test.json'), classification=True)


                print("Train size: ", df_train.shape, "Valid size: ", df_valid.shape, "Test size: ", df_test.shape)
                print()

                total_pred, total_true, num_parameters = train_test(df_train, df_valid, df_test, args, stemmer, sp, folder)

            p_5, r_5, f_5, p_10, r_10, f_10, p_k, r_k, f_k, p_M, r_M, f_M = eval(total_pred, total_true)

            result_file.write("Dataset: " + folder + '\n')
            result_file.write('Precision@5: ' + str(p_5) + ' Recall@5: ' + str(r_5) + ' F1@5: ' + str(f_5) + '\n')
            result_file.write('Precision@10: ' + str(p_10) + ' Recall@10: ' + str(r_10) + ' F1@10: ' + str(f_10) + '\n')
            result_file.write('Precision@k: ' + str(p_k) + ' Recall@k: ' + str(r_k) + ' F1@k: ' + str(f_k) + '\n')
            result_file.write('Precision@M: ' + str(p_M) + ' Recall@M: ' + str(r_M) + ' F1@M: ' + str(f_M) + '\n')
            result_file.write('Num. trainable parameters: ' + str(num_parameters) + '\n')

            outputs = []

            for pred, true in zip(total_pred, total_true):
                pred = ";".join(list(pred))
                true = ";".join(list(true))
                outputs.append((pred, true))

            df_preds = pd.DataFrame(outputs, columns=['Predicted', 'True'])
            df_preds.to_csv(os.path.join(args.result_path, 'predictions', folder + '.csv'), sep=',', encoding='utf8')

        result_file.write("\n-----------------------------------------------------------\n")
        result_file.write("\n-----------------------End of the run----------------------\n")
        result_file.write("\n-----------------------------------------------------------\n")
        result_file.close()



def train_test(df_train, df_valid, df_test, args, stemmer, sp, folder=None):
    print('Producing dataset...')
    corpus = Corpus(df_train, df_valid, df_test, args)

    print()
    print('Batchifying')

    if not args.classification:
        train_data = batchify(corpus.train, args.batch_size, args.n_ctx)
        val_data = batchify(corpus.valid, args.batch_size, args.n_ctx)
        test_data = batchify(corpus.test, args.batch_size, args.n_ctx)
        if args.POS_tags:
            train_pos = batchify(corpus.train_pos, args.batch_size, args.n_ctx)
            val_pos = batchify(corpus.valid_pos, args.batch_size, args.n_ctx)
            test_pos = batchify(corpus.test_pos, args.batch_size, args.n_ctx)

        val_target = None
        valid_keywords = None
        test_target = None
        test_keywords = None
    else:
        valid_keywords = corpus.valid_keywords
        test_keywords = corpus.test_keywords

        train_data, train_target = batchify_docs(corpus.train, corpus.train_target, args.batch_size)
        val_data, val_target = batchify_docs(corpus.valid, corpus.valid_target, args.batch_size)
        test_data, test_target = batchify_docs(corpus.test, corpus.test_target, 1)
        if args.POS_tags:
            train_pos, _ = batchify_docs(corpus.train_pos, corpus.train_target, args.batch_size)
            val_pos, _ = batchify_docs(corpus.valid_pos, corpus.valid_target, args.batch_size)
            test_pos, _ = batchify_docs(corpus.test_pos, corpus.test_target, 1)

    ntokens = len(corpus.dictionary)
    print('Vocabulary size: ', ntokens)
    args.vocab_size = ntokens

    # adaptive softmax / embedding
    cutoffs, tie_projs = [], [False]
    print("Adaptive softmax: ", args.adaptive)
    if args.adaptive:
        if not args.bpe:
            cutoffs = [20000, 40000, 200000]
        else:
            cutoffs = [20000, 30000]
        tie_projs += [True] * len(cutoffs)

    args.cutoffs = cutoffs
    args.tie_projs = tie_projs

    if args.classification and args.transfer_learning:
        model = torch.load(args.language_model_path)
        model.head = TransformerHead(model.wte, args)
        model.config = args
    elif args.transfer_learning:
        print('Domain adaptation language modelling')
        model = torch.load(args.language_model_path)
        model.config = args
    else:
        model = TransformerModel(args)
    if args.cuda:
        model.cuda()
    else:
        model.cpu()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step, eta_min=args.eta_min)


    best_loss = 9999
    best_f = 0
    best_model_path = ''

    train_step = 0

    for epoch in range(args.num_epoch):
        print()
        print("Epoch: ", epoch + 1, "Num. train batches: ", train_data.size(1))
        print()

        model.train()

        total_loss = 0
        total_seq = 0

        i = 0
        cut = 0
        if not args.classification:
            cut = args.n_ctx
            all_steps = train_data.size(1)
        else:
            all_steps = train_data.size(0)


        while i < all_steps - cut:

            if not args.classification:
                encoder_words, batch_labels, mask = get_batch(train_data, i, args, corpus.dictionary.word2idx)
                if args.POS_tags:
                    encoder_pos, _, _= get_batch(train_pos, i, args, corpus.dictionary.word2idx, mask)

            else:
                encoder_words, batch_labels = get_batch_docs(train_data, train_target, i, args)

                if args.POS_tags:
                    encoder_pos, _ = get_batch_docs(train_pos, train_target, i, args)

                mask = None

            if not args.POS_tags:
                encoder_pos=None

            optimizer.zero_grad()
            flat_indexed_words = encoder_words.contiguous().view(-1)


            loss = model(encoder_words, input_pos=encoder_pos, lm_labels=batch_labels, masked_idx=mask)
            loss = loss.float().mean().type_as(loss)
            loss.backward()

            # `clip_grad_norm`
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            train_step += 1
            scheduler.step()
            if args.classification:
                report_step = 32
            else:
                report_step = 10240
            if train_step % report_step == 0:
                print("Learning rate: ", optimizer.param_groups[0]['lr'])

            if not args.classification:
                i += args.n_ctx
                total_loss += batch_labels.size(0) * loss.item()
                total_seq += batch_labels.size(0)
            else:
                i += 1
                total_loss +=  loss.item()
                total_seq += 1

            if i % report_step == 0:
                print('Step: ', i, ' loss: ', total_loss/total_seq)


        #Validation
        print()
        print('Validating')
        print()
        if not args.POS_tags:
            val_pos = None


        total_loss, total_seq, total_pred, total_true = test(model, val_data, val_pos, val_target, corpus, args, stemmer, valid_keywords, sp)
        total_loss = total_loss/total_seq

        if not args.classification:
            perplexity = math.exp(total_loss)
            score = str(perplexity)[:6]
            print("Validation loss: ", total_loss)
            print("Validation set perplexity: ", perplexity)

            if  total_loss < best_loss:
                path = os.path.join(args.trained_language_models_dir, "model_" + args.config_id + "_perp_" + score + "_epoch_" + str(epoch + 1) + ".pt")
                with open(path, 'wb') as f:
                    print('Saving model')
                    torch.save(model, f)

                #delete all models but the best
                if best_model_path:
                    if os.path.isfile(best_model_path):
                        os.remove(best_model_path)

                best_model_path = path
                best_loss = total_loss
        else:
            print('Validating on ', folder)
            p_5, r_5, f_5, p_10, r_10, f_10, p_k, r_k, f_k, p_M, r_M, f_M = eval(total_pred, total_true)
            score = str(total_loss)

            if f_10 > best_f:
                path = os.path.join(args.trained_classification_models_dir, "model_" + args.config_id + "_folder_" + folder +  "_loss_" + score + "_epoch_" + str(epoch + 1) + ".pt")
                with open(path, 'wb') as f:
                    print('Saving model')
                    torch.save(model, f)

                # delete all models but the best
                if best_model_path:
                    if os.path.isfile(best_model_path):
                        os.remove(best_model_path)
                best_model_path = path
                best_f = f_10

        gc.collect()

    del model
    del optimizer
    del scheduler

    model = torch.load(best_model_path)
    num_parameters = str(count_parameters(model))

    print()
    print('Testing on test set')
    print()

    if not args.POS_tags:
        test_pos = None

    total_loss, total_seq, total_pred, total_true = test(model, test_data, test_pos, test_target, corpus, args, stemmer, test_keywords, sp)
    total_loss = total_loss / total_seq

    gc.collect()
    del model


    if not args.classification:
        perplexity = math.exp(total_loss)
        print("Test loss: ", total_loss)
        print("Test set perplexity: ", perplexity)
        return None

    else:
        print()
        print('------------------------------------------------------------------------------------------------------------------')
        print()
        print('Testing on ', folder)

        return total_pred, total_true, num_parameters



def test(model, data, data_pos, target, corpus, args, stemmer, keywords=None, sp=None):
    # testing

    total_pred = []
    total_true = []
    total_loss = 0
    total_seq = 0

    model.eval()
    if not args.classification:
        step = args.n_ctx
        cut = args.n_ctx
        all_steps = data.size(1)
    else:
        step = 1
        cut = 0
        all_steps = data.size(0)

    if not args.POS_tags:
        encoder_pos=None

    with torch.no_grad():

        all_predicted_save = []
        all_true_save = []

        for i in range(0, all_steps - cut, step):

            if not args.classification:
                encoder_words, batch_labels, mask = get_batch(data, i, args, corpus.dictionary.word2idx)
                if args.POS_tags:
                    encoder_pos, _, _ = get_batch(data_pos, i, args, corpus.dictionary.word2idx, mask)
            else:
                encoder_words, batch_labels = get_batch_docs(data, target, i, args)
                if args.POS_tags:
                    encoder_pos, _ = get_batch_docs(data_pos, target, i, args)
                mask = None


            input_batch_labels = batch_labels.clone()

            if not args.classification:
                loss, logits = model(encoder_words, input_pos=encoder_pos, lm_labels=input_batch_labels, masked_idx=mask, test=True)
            else:
                if not args.crf:
                    loss, logits, att_vector = model(encoder_words, input_pos=encoder_pos, lm_labels=input_batch_labels, test=True)
                else:
                    loss, logits, crf_preds, att_vector = model(encoder_words, input_pos=encoder_pos, lm_labels=input_batch_labels, test=True)

            loss = loss.mean()
            total_loss += batch_labels.size(0) * loss.float().item()
            total_seq += batch_labels.size(0)

            if args.classification:
                report_step = 32
            else:
                report_step = 10240

            if i % report_step == 0:
                print('Eval step: ', i, 'Loss: ', total_loss/total_seq)

            if args.classification:

                maxes = []
                true_y = []

                for batch in encoder_words.cpu().numpy():
                    key = "".join([str(x) for x in batch if x != 0])
                    true_example = keywords[key]
                    true_example = [" ".join(kw) for kw in true_example]
                    true_y.append(true_example)
                    all_true_save.append(true_example)

                batch_counter = 0
                for batch_idx, batch in enumerate(logits):

                    pred_save = []

                    pred_example = []
                    batch = F.softmax(batch, dim=1)
                    length = batch.size(0)
                    position = 0

                    pred_vector = []
                    probs_dict = {}

                    while position < len(batch):
                        pred = batch[position]
                        if not args.crf:
                            _ , idx = pred.max(0)
                            idx = idx.item()
                        else:
                            idx = crf_preds[batch_idx][position]

                        pred_vector.append(pred)
                        pred_word = []

                        if idx == 1:
                            words = []
                            num_steps = length - position
                            for j in range(num_steps):
                                new_pred = batch[position + j]
                                values, new_idx = new_pred.max(0)

                                if not args.crf:
                                    new_idx = new_idx.item()
                                else:
                                    new_idx = crf_preds[batch_idx][position + j]
                                prob = values.item()

                                if new_idx == 1:
                                    word = corpus.dictionary.idx2word[encoder_words[batch_counter][position + j]]
                                    words.append((word, prob))
                                    pred_word.append((word, prob))
                                    stem = stemmer.stem(word)

                                    if stem not in probs_dict:
                                        probs_dict[stem] = prob
                                    else:
                                        if probs_dict[stem] < prob:
                                            probs_dict[stem] = prob
                                else:
                                    if sp is not None:
                                        word = corpus.dictionary.idx2word[encoder_words[batch_counter][position + j]]
                                        if not word.startswith('▁'):
                                            words.append((word, prob))

                                            stem = stemmer.stem(word)
                                            if stem not in probs_dict:
                                                probs_dict[stem] = prob
                                            else:
                                                if probs_dict[stem] < prob:
                                                    probs_dict[stem] = prob
                                    break

                            position += j + 1
                            words = [x[0] for x in words]
                            if sp is not None:
                                if words[0].startswith('▁'):
                                    pred_example.append(words)
                                    pred_save.append(pred_word)
                            else:
                                pred_example.append(words)
                                pred_save.append(pred_word)
                        else:
                            position += 1

                    all_predicted_save.append(pred_save)


                    #assign probabilities
                    pred_examples_with_probs = []
                    for kw in pred_example:
                        probs = []
                        for word in kw:
                            stem = stemmer.stem(word)
                            probs.append(probs_dict[stem])

                        kw_prob = sum(probs)/len(probs)
                        pred_examples_with_probs.append((" ".join(kw), kw_prob))

                    pred_example = pred_examples_with_probs

                    #sort by softmax probability
                    pred_example = sorted(pred_example, reverse=True, key=lambda x: x[1])

                    #remove keywords that contain punctuation and duplicates
                    all_kw = set()
                    filtered_pred_example = []
                    kw_stems = []

                    punctuation = "!#$%&'()*+,.:;<=>?@[\]^_`{|}~"

                    for kw, prob in pred_example:
                        if sp is not None:
                            kw_decoded = sp.DecodePieces(kw.split())
                            kw_stem = " ".join([stemmer.stem(word) for word in kw_decoded.split()])
                        else:
                            kw_stem = " ".join([stemmer.stem(word) for word in kw.split()])
                        kw_stems.append(kw_stem)

                        if kw_stem not in all_kw and len(kw_stem.split()) == len(set(kw_stem.split())):
                            has_punct = False
                            for punct in punctuation:
                                if punct in kw:
                                    has_punct = True
                                    break
                            if sp is not None:
                                kw_decoded = sp.DecodePieces(kw.split())
                                if not has_punct and len(kw_decoded.split()) < 5:
                                    filtered_pred_example.append((kw, prob))
                            else:
                                if not has_punct and len(kw.split()) < 5:
                                    filtered_pred_example.append((kw, prob))
                        all_kw.add(kw_stem)

                    pred_example = filtered_pred_example
                    filtered_pred_example = [x[0] for x in pred_example][:args.kw_cut]

                    maxes.append(filtered_pred_example)
                    batch_counter += 1

                if sp is not None:
                    all_decoded_maxes = []
                    all_decoded_true_y = []
                    for doc in maxes:
                        decoded_maxes = []
                        for kw in doc:
                            kw = sp.DecodePieces(kw.split())
                            decoded_maxes.append(kw)
                        all_decoded_maxes.append(decoded_maxes)
                    for doc in true_y:
                        decoded_true_y = []
                        for kw in doc:
                            kw = sp.DecodePieces(kw.split())
                            decoded_true_y.append(kw)
                        all_decoded_true_y.append(decoded_true_y)

                    maxes = all_decoded_maxes
                    true_y = all_decoded_true_y

                total_pred.extend(maxes)
                total_true.extend(true_y)

    return total_loss, total_seq, total_pred, total_true


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_id', type=str, required=True, help='Used to connect trained language models with classification models')
    parser.add_argument('--datasets', type=str, help='Names of dataset folders divided by ; - should be in the data directory')
    parser.add_argument('--lm_corpus_file', type=str, help='Name of language model train corpus -should be in the data directory')
    parser.add_argument('--bpe_model_path', type=str, help='Path to trained byte pair encoding model')
    parser.add_argument('--data_path', type=str, default='data', help='Path to directory containing all corpora')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--trained_language_models_dir', type=str, default='trained_language_models')
    parser.add_argument('--trained_classification_models_dir', type=str, default='trained_classification_models')
    parser.add_argument('--dict_folder', type=str, default='dictionaries', help='Path to dictionary folder')
    parser.add_argument('--cv_eval', type=str, default='nus;duc', help='Names of dataset folders on which cross validation should be conducted divided by ;')
    parser.add_argument('--kw_cut', type=int, default=10, help='Max number of returned keywords')

    parser.add_argument('--cuda', action='store_true', help='If true, unconditional generation.')

    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--layer_norm_epsilon", type=float, default=1e-6)
    parser.add_argument("--initializer_range", type=float, default=0.02)

    parser.add_argument("--n_ctx", type=int, default=512)
    parser.add_argument("--n_positions", type=int, default=512)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--max_vocab_size", type=int, default=0, help='Zero means no limit.')

    parser.add_argument('--max_step', type=int, default=100000, help='upper step limit used in cosine scheduler')
    parser.add_argument('--eta_min', type=float, default=0.0, help='min learning rate for cosine scheduler')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')

    parser.add_argument("--num_epoch", type=int, default=10)

    parser.add_argument('--adaptive', action='store_true', help='If true, use adaptive softmax.')
    parser.add_argument('--bpe', action='store_true', help='If true, use byte pair encoding.')
    parser.add_argument('--masked_lm', action='store_true', help='If true, use masked language model objective for pretraining instead of regular language model.')
    parser.add_argument('--transfer_learning', action='store_true', help='If true, use a pretrained language model.')
    parser.add_argument('--POS_tags', action='store_true', help='If true, use additional POS tag sequence input')
    parser.add_argument('--classification', action='store_true', help='If true, train a classifier.')
    parser.add_argument('--rnn', action='store_true', help='If true, use a RNN with attention in classification head.')
    parser.add_argument('--crf', action='store_true', help='If true, use a BiLSTM-CRF token classification head.')

    args = parser.parse_args()

    run_model(args)

































