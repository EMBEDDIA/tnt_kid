import sentencepiece as spm
from preprocessing import file_to_df
import os
from nltk import sent_tokenize
import argparse


def train_bpe_model(input, output):
    df = file_to_df(os.path.join('data/' + input), classification=False)
    with open('data/' + output + '.txt', 'w', encoding='utf8') as f:
        for idx, line in df.iterrows():
            text = line['text']
            sents = sent_tokenize(text)
            for sent in sents:
                f.write(sent.lower().strip() + '\n')

    assert not os.path.exists(output + '.model')

    spm.SentencePieceTrainer.Train('--input=data/' + output + '.txt --model_prefix=' + output + ' --vocab_size=32000 --character_coverage=1.0')

    sp = spm.SentencePieceProcessor()
    sp.Load(output + ".model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data_news.json')
    parser.add_argument('--output', type=str, default='bpe_news')
    args = parser.parse_args()
    train_bpe_model(args.input, args.output)
