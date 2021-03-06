# Code for experiments conducted in the paper 'TNT-KID: Transformer-based Neural Tagger for Keyword Identification' submitted to Natural Language Engineering journal #

Please cite the following paper [[bib](https://gitlab.com/matej.martinc/tnt_kid/-/blob/master/bibtex.js)] if you use this code:

Matej Martinc, Blaz Skrlj and Senja Pollak. TNT-KID: Transformer-based Neural Tagger for Keyword Identification. arXiv preprint arXiv:2003.09166.


## Installation, documentation ##

Published results were produced in Python 3 programming environment on Linux Mint 18 Cinnamon operating system. Instructions for installation assume the usage of PyPI package manager.<br/>
To get the source code and the train and test data, clone the project from the repository with 'git clone https://gitlab.com/matej.martinc/tnt_kid'<br/>
To only get the source code, clone the repository from github with 'git clone https://github.com/EMBEDDIA/tnt_kid'<br/>

Install dependencies if needed: pip install -r requirements.txt

### To extract keywords with the model trained on scientific texts (KP20k dataset): ###

```
python predict.py --data_path data/example.json --bpe_model_path bpe/bpe_science.model --dict_path dictionaries/5_lm+bpe+rnn_science_adaptive_lm_bpe_nopos_rnn_nocrf.ptb --trained_classification_model trained_classification_models/model_5_lm+bpe+rnn_science_folder_kp20k.pt --adaptive --rnn --bpe --cuda
```

### To reproduce the results published in the paper run the code in the command line using following commands: ###

Generate news and science datasets for language model training:<br/>
```
python data/build_dataset.py --datasets data --data_path data
```

Train language model on the computer science domain articles:<br/>
```
python train_and_eval.py --config_id 5_lm+bpe+rnn_science --lm_corpus_file data_science.json --bpe_model_path bpe/bpe_science.model --adaptive --rnn --bpe --cuda
```

Train language model on the news articles:<br/>
```
python train_and_eval.py --config_id 5_lm+bpe+rnn_news --lm_corpus_file data_news.json --bpe_model_path bpe/bpe_news.model --adaptive --rnn --bpe --cuda
```

Train and test the keyword tagger on the datasets from the computer science domain:<br/>
```
python train_and_eval.py --config_id 5_lm+bpe+rnn_science --bpe_model_path bpe/bpe_science.model --datasets 'kp20k;inspec;krapivin;nus;semeval' --classification --transfer_learning --rnn --bpe --cuda
```

Train and test the keyword tagger on the datasets from the news domain:<br/>
```
python train_and_eval.py --config_id 5_lm+bpe+rnn_news --bpe_model_path bpe/bpe_news.model --datasets 'kptimes;jptimes;duc' --classification --transfer_learning --rnn --bpe --cuda
```

## Contributors to the code ##

Matej Martinc<br/>

* [Knowledge Technologies Department](http://kt.ijs.si), Jožef Stefan Institute, Ljubljana
