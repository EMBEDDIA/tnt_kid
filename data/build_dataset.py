import os
import argparse

def build_science_dataset(args):
    folders = ['inspec', 'kp20k', 'krapivin', 'nus', 'semeval']
    output_all_data = open(os.path.join(args.data_path, 'data_science.json'), 'w', encoding='utf8')

    for fold in folders:
        test_path = os.path.join(args.datasets, fold, fold + '_test.json')
        valid_path = os.path.join(args.datasets, fold, fold + '_valid.json')

        counter = 0

        with open(test_path, 'r', encoding='utf8') as f:
            for line in f:
                output_all_data.write(line)
                counter += 1

        if fold != 'nus':
            with open(valid_path, 'r', encoding='utf8') as f:
                for line in f:
                    output_all_data.write(line)
                    counter += 1

        if fold == 'kp20k':
            train_path = os.path.join(args.datasets, fold, fold + '_train.json')
            with open(train_path, 'r', encoding='utf8') as f:
                for line in f:
                    output_all_data.write(line)
                    counter += 1

        print("Done with ", fold, ", Num. docs: ", counter)



    output_all_data.close()


def build_news_dataset(args):
    folders = ['duc', 'kptimes', 'jptimes']

    output_all_data = open(os.path.join(args.data_path, 'data_news.json'), 'w', encoding='utf8')

    for fold in folders:
        test_path = os.path.join(args.datasets, fold, fold + '_test.json')
        valid_path = os.path.join(args.datasets, fold, fold + '_valid.json')

        counter = 0

        with open(test_path, 'r', encoding='utf8') as f:
            for line in f:
                output_all_data.write(line)
                counter += 1

        if fold != 'duc':
            with open(valid_path, 'r', encoding='utf8') as f:
                for line in f:
                    output_all_data.write(line)
                    counter += 1

        if fold == 'kptimes':
            train_path = os.path.join(args.datasets, fold, fold + '_train.json')
            with open(train_path, 'r', encoding='utf8') as f:
                for line in f:
                    output_all_data.write(line)
                    counter += 1

        print("Done with ", fold, ", Num. docs: ", counter)

    output_all_data.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, help='Path to input folder with input datasets', default='data')
    parser.add_argument('--data_path', type=str, default='data', help='Path to output directory containing the language model dataset')


    args = parser.parse_args()
    build_science_dataset(args)
    build_news_dataset(args)





