import os

def build_science_dataset():
    folders = ['inspec', 'kp20k', 'krapivin', 'nus', 'semeval']
    output_all_data = open('data_science.json', 'w', encoding='utf8')

    for fold in folders:
        test_path = os.path.join(fold, fold + '_test.json')
        valid_path = os.path.join(fold, fold + '_valid.json')

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
            train_path = os.path.join(fold, fold + '_train.json')
            with open(train_path, 'r', encoding='utf8') as f:
                for line in f:
                    output_all_data.write(line)
                    counter += 1

        print("Done with ", fold, ", Num. docs: ", counter)



    output_all_data.close()


def build_news_dataset():
    folders = ['duc', 'kptimes', 'jptimes']

    output_all_data = open('data_news.json', 'w', encoding='utf8')

    for fold in folders:
        test_path = os.path.join(fold, fold + '_test.json')
        valid_path = os.path.join(fold, fold + '_valid.json')

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
            train_path = os.path.join(fold, fold + '_train.json')
            with open(train_path, 'r', encoding='utf8') as f:
                for line in f:
                    output_all_data.write(line)
                    counter += 1

        print("Done with ", fold, ", Num. docs: ", counter)

    output_all_data.close()


if __name__ == '__main__':
    build_science_dataset()
    build_news_dataset()





