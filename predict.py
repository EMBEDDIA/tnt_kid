import argparse
import torch
import pandas as pd
import pickle
import sentencepiece as spm
import json
from nltk import sent_tokenize, word_tokenize
import torch.nn.functional as F
from nltk.stem.porter import *


def file_to_df(data_path):
    all_docs = []
    counter = 0
    num_words = 0
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f:
            counter += 1
            if counter % 10000 == 0:
                print('Processing json: ', counter)
            line = json.loads(line)
            title = line.get('title') or ''
            abstract = line.get('abstract') or ''
            text = title + '. ' + abstract
            num_words += len(text.split())
            all_docs.append([text])

    df = pd.DataFrame(all_docs)
    df.columns = ["text"]
    print(data_path, 'data size: ', df.shape)
    print('Avg words: ', num_words / df.shape[0])
    return df



class Corpus(object):
    def __init__(self, df_test, args):

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.bpe_model_path)
        self.max_length = args.n_ctx

        with open(args.dict_path, 'rb') as file:
            self.dictionary = pickle.load(file)

        self.test = self.tokenize_doc(df_test, max_length=self.max_length)


    def preprocess_line(self, line):
        words = []
        text = line['text']
        text = text.replace('-', ' ')
        text = text.replace('/', ' ')
        text = text.replace('∗', ' ')
        for sent in sent_tokenize(text):
            sent = word_tokenize(sent)

            bpe_sent = []
            for w in sent:
                w = w.lower()
                bpe_word = self.sp.EncodeAsPieces(w)
                bpe_sent.append(bpe_word)
                words.extend(bpe_word)
            words.append('<eos>')
        return words


    def tokenize_doc(self, df, max_length):
        x = torch.zeros([df.shape[0], max_length], dtype=torch.long)
        i = 0
        for _, line in df.iterrows():

            words = self.preprocess_line(line)

            for j, word in enumerate(words):
                if word in self.dictionary.word2idx:
                    idx = self.dictionary.word2idx[word]
                else:
                    idx = self.dictionary.word2idx['<unk>']
                if j < max_length:
                    x[i][j] = idx
            i += 1
        return x


def batchify_docs(data, bsz):
    nbatch = data.size(0) // bsz
    doc_length = data.size(1)
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(-1, bsz, doc_length).contiguous()
    return data


def get_batch_docs(data, i, config):
    if config.cuda:
        return data[i, :, :].cuda()
    return data[i, :, :]


def loadModel(model_path, args):
    if not args.cuda:
        kw_model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        kw_model = torch.load(model_path)
    kw_model.config.cuda = args.cuda
    if args.cuda:
        kw_model.cuda()
    else:
        kw_model.cpu()
    return kw_model


def predict(test_data, model, stemmer, sp):
    step = 1
    cut = 0
    all_steps = test_data.size(0)
    encoder_pos = None
    total_pred = []

    with torch.no_grad():

        for i in range(0, all_steps - cut, step):

            encoder_words = get_batch_docs(test_data, i, args)
            logits = model(encoder_words, input_pos=encoder_pos, lm_labels=None, predict=True)
            maxes = []

            batch_counter = 0

            for batch in logits:

                pred_example = []
                batch = F.softmax(batch, dim=1)
                length = batch.size(0)
                position = 0
                probs_dict = {}

                while position < len(batch):
                    pred = batch[position]

                    _, idx = pred.max(0)
                    idx = idx.item()

                    if idx == 1:
                        words = []
                        num_steps = length - position
                        for j in range(num_steps):
                            new_pred = batch[position + j]
                            values, new_idx = new_pred.max(0)
                            new_idx = new_idx.item()
                            prob = values.item()

                            if new_idx == 1:
                                word = corpus.dictionary.idx2word[encoder_words[batch_counter][position + j]]
                                words.append((word, prob))

                                # add max word prob in document to prob dictionary
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
                                        words = []
                                break

                        position += j + 1
                        words = [x[0] for x in words]
                        if sp is not None:
                            if len(words) > 0 and words[0].startswith('▁'):
                                pred_example.append(words)
                        else:
                            pred_example.append(words)
                    else:
                        position += 1

                # assign probabilities
                pred_examples_with_probs = []
                for kw in pred_example:
                    probs = []
                    for word in kw:
                        stem = stemmer.stem(word)
                        probs.append(probs_dict[stem])

                    kw_prob = sum(probs) / len(probs)
                    pred_examples_with_probs.append((" ".join(kw), kw_prob))

                pred_example = pred_examples_with_probs

                # sort by softmax probability
                pred_example = sorted(pred_example, reverse=True, key=lambda x: x[1])

                # remove keywords that contain punctuation and duplicates
                all_kw = set()
                filtered_pred_example = []
                kw_stems = []

                punctuation = "!#$%&'()*+,.:;<=>?@[\]^_`{|}~"

                for kw, prob in pred_example:

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
                for doc in maxes:
                    decoded_maxes = []
                    for kw in doc:
                        kw = sp.DecodePieces(kw.split())
                        decoded_maxes.append(kw)
                    all_decoded_maxes.append(decoded_maxes)

                maxes = all_decoded_maxes

            total_pred.extend(maxes)
    return total_pred



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to data in json form')
    parser.add_argument('--bpe_model_path', type=str, help='Path to trained byte pair encoding model')
    parser.add_argument('--trained_classification_model', type=str, help='Path to pretrained classification model')
    parser.add_argument('--dict_path', type=str, help='Path to dictionary')
    parser.add_argument('--result_path', type=str, default='results/predictions.csv')

    parser.add_argument('--kw_cut', type=int, default=10, help='Max number of returned keywords')
    parser.add_argument('--cuda', action='store_false', help='If true, unconditional generation.')
    parser.add_argument("--seed", type=int, default=2019)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_ctx", type=int, default=256)
    parser.add_argument("--n_positions", type=int, default=256)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--max_vocab_size", type=int, default=0, help='Zero means no limit.')

    parser.add_argument('--adaptive', action='store_true', help='If true, use adaptive softmax.')
    parser.add_argument('--bpe', action='store_true', help='If true, use byte pair encoding.')
    parser.add_argument('--masked_lm', action='store_true',
                        help='If true, use masked language model objective for pretraining instead of regular language model.')
    parser.add_argument('--transfer_learning', action='store_true', help='If true, use a pretrained language model.')
    parser.add_argument('--POS_tags', action='store_true', help='If true, use additional POS tag sequence input')
    parser.add_argument('--classification', action='store_true', help='If true, train a classifier.')
    parser.add_argument('--rnn', action='store_true', help='If true, use a RNN with attention in classification head.')
    parser.add_argument('--crf', action='store_true', help='If true, use a BiLSTM-CRF token classification head.')
    args = parser.parse_args()

    df_test = file_to_df(args.data_path)
    corpus = Corpus(df_test, args)
    test_data = batchify_docs(corpus.test, 1)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.bpe_model_path)

    stemmer = PorterStemmer()

    model = loadModel(args.trained_classification_model, args)
    model.eval()


    predictions = predict(test_data, model, stemmer, sp)
    predictions = [";".join(kws) for kws in predictions]
    df_kw = pd.DataFrame(predictions, columns=['keywords'])
    df = pd.concat([df_test, df_kw], axis=1)
    df.to_csv(args.result_path, encoding='utf8', sep='\t')



