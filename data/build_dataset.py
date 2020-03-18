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

        with open(valid_path, 'r', encoding='utf8') as f:
            for line in f:
                output_all_data.write(line)
                counter += 1

        print(fold, counter)

    output_all_data.close()


def build_news_dataset():
    folders = ['duc', 'kptimes']

    output_all_data = open('data_news.json', 'w', encoding='utf8')

    for fold in folders:
        test_path = os.path.join(fold, fold + '_test.json')
        valid_path = os.path.join(fold, fold + '_valid.json')

        counter = 0

        with open(test_path, 'r', encoding='utf8') as f:
            for line in f:
                output_all_data.write(line)
                counter += 1

        if fold == 'kptimes':
            with open(valid_path, 'r', encoding='utf8') as f:
                for line in f:
                    output_all_data.write(line)
                    counter += 1

        print(fold, counter)

    output_all_data.close()


def split_kptimes():
    folders = ['kptimes']

    output_all_data = open('kptimes_test.json', 'w', encoding='utf8')

    for fold in folders:
        test_path = os.path.join(fold, fold + '_test.json')

        counter = 0

        with open(test_path, 'r', encoding='utf8') as f:
            for line in f:
                print(counter)
                if counter >= 10000:
                    output_all_data.write(line)
                counter += 1



        print(fold, counter)

    output_all_data.close()



if __name__ == '__main__':
    build_science_dataset()
    build_news_dataset()





