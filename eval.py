import scipy
from nltk.stem.porter import *
import numpy as np
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu as bleu

stemmer = PorterStemmer()


def stem_word_list(word_list):
    return [stemmer.stem(w.strip()) for w in word_list]


def macro_averaged_score(precisionlist, recalllist):
    precision = np.average(precisionlist)
    recall = np.average(recalllist)
    f_score = 0
    if (precision or recall):
        f_score = round((2 * (precision * recall)) / (precision + recall), 4)
    return precision, recall, f_score


def get_match_result(true_seqs, pred_seqs, do_stem=True, type='exact'):
    '''
    If type='exact', returns a list of booleans indicating if a pred has a matching tgt
    If type='partial', returns a 2D matrix, each value v_ij is a float in range of [0,1]
        indicating the (jaccard) similarity between pred_i and tgt_j
    :param true_seqs:
    :param pred_seqs:
    :param do_stem:
    :param topn:
    :param type: 'exact' or 'partial'
    :return:
    '''
    # do processing to baseline predictions
    if type == "exact":
        match_score = np.zeros(shape=(len(pred_seqs)), dtype='float32')
    else:
        match_score = np.zeros(shape=(len(pred_seqs), len(true_seqs)), dtype='float32')

    target_number = len(true_seqs)
    predicted_number = len(pred_seqs)

    metric_dict = {'target_number': target_number, 'prediction_number': predicted_number, 'correct_number': match_score}

    # convert target index into string
    if do_stem:
        true_seqs = [stem_word_list(seq) for seq in true_seqs]
        pred_seqs = [stem_word_list(seq) for seq in pred_seqs]

    for pred_id, pred_seq in enumerate(pred_seqs):
        if type == 'exact':
            match_score[pred_id] = 0
            for true_id, true_seq in enumerate(true_seqs):
                match = True
                if len(pred_seq) != len(true_seq):
                    continue
                for pred_w, true_w in zip(pred_seq, true_seq):
                    # if one two words are not same, match fails
                    if pred_w != true_w:
                        match = False
                        break
                # if every word in pred_seq matches one true_seq exactly, match succeeds
                if match:
                    match_score[pred_id] = 1
                    break
        elif type == 'ngram':
            # use jaccard coefficient as the similarity of partial match (1+2 grams)
            pred_seq_set = set(pred_seq)
            pred_seq_set.update(set([pred_seq[i] + '_' + pred_seq[i + 1] for i in range(len(pred_seq) - 1)]))
            for true_id, true_seq in enumerate(true_seqs):
                true_seq_set = set(true_seq)
                true_seq_set.update(set([true_seq[i] + '_' + true_seq[i + 1] for i in range(len(true_seq) - 1)]))
                if float(len(set.union(*[set(true_seq_set), set(pred_seq_set)]))) > 0:
                    similarity = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) \
                                 / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                else:
                    similarity = 0.0
                match_score[pred_id, true_id] = similarity
        elif type == 'mixed':
            # similar to jaccard, but addtional to 1+2 grams we also put in the full string, serves like a exact+partial surrogate
            pred_seq_set = set(pred_seq)
            pred_seq_set.update(set([pred_seq[i] + '_' + pred_seq[i + 1] for i in range(len(pred_seq) - 1)]))
            pred_seq_set.update(set(['_'.join(pred_seq)]))
            for true_id, true_seq in enumerate(true_seqs):
                true_seq_set = set(true_seq)
                true_seq_set.update(set([true_seq[i] + '_' + true_seq[i + 1] for i in range(len(true_seq) - 1)]))
                true_seq_set.update(set(['_'.join(true_seq)]))
                if float(len(set.union(*[set(true_seq_set), set(pred_seq_set)]))) > 0:
                    similarity = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) \
                                 / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
                else:
                    similarity = 0.0
                match_score[pred_id, true_id] = similarity

        elif type == 'bleu':
            # account for the match of subsequences, like n-gram-based (BLEU) or LCS-based
            # n-gras precision doesn't work that well
            for true_id, true_seq in enumerate(true_seqs):
                match_score[pred_id, true_id] = bleu(pred_seq, [true_seq], [0.7, 0.3, 0.0])

    return match_score


def run_metrics(match_list, pred_list, tgt_list, score_names, topk_range, type='exact'):
    """
    Return a dict of scores containing len(score_names) * len(topk_range) items
    score_names and topk_range actually only define the names of each score in score_dict.
    :param match_list:
    :param pred_list:
    :param tgt_list:
    :param score_names:
    :param topk_range:
    :return:
    """
    score_dict = {}
    if len(tgt_list) == 0:
        for topk in topk_range:
            for score_name in score_names:
                score_dict['{}@{}'.format(score_name, topk)] = 0.0
        return score_dict

    assert len(match_list) == len(pred_list)
    for topk in topk_range:
        if topk == 'k':
            cutoff = len(tgt_list)
        elif topk == 'M':
            cutoff = len(pred_list)
        else:
            cutoff = topk

        if len(pred_list) > cutoff:
            pred_list_k = np.asarray(pred_list[:cutoff])
            match_list_k = match_list[:cutoff]
        else:
            pred_list_k = np.asarray(pred_list)
            match_list_k = match_list


        # Micro-Averaged Method
        correct_num = int(sum(match_list_k))
        # Precision, Recall and F-score, with flexible cutoff (if number of pred is smaller)
        micro_p = float(sum(match_list_k)) / float(len(pred_list_k)) if len(pred_list_k) > 0 else 0.0
        micro_r = float(sum(match_list_k)) / float(len(tgt_list)) if len(tgt_list) > 0 else 0.0


        if micro_p + micro_r > 0:
            micro_f1 = float(2 * (micro_p * micro_r)) / (micro_p + micro_r)
        else:
            micro_f1 = 0.0
        # F-score, with a hard cutoff on precision, offset the favor towards fewer preds
        micro_p_hard = float(sum(match_list_k)) / cutoff if len(pred_list_k) > 0 else 0.0
        if micro_p_hard + micro_r > 0:
            micro_f1_hard = float(2 * (micro_p_hard * micro_r)) / (micro_p_hard + micro_r)
        else:
            micro_f1_hard = 0.0

        for score_name, v in zip(score_names, [correct_num, micro_p, micro_r, micro_f1, micro_p_hard, micro_f1_hard]):
            score_dict['{}@{}'.format(score_name, topk)] = v

    return score_dict


def eval(detected_keywords, gold_standard_keywords, remove_docs_with_no_kw=True):
    all_p_et_5 = []
    all_r_et_5 = []
    all_p_et_10 = []
    all_r_et_10 = []
    all_p_et_k = []
    all_r_et_k = []
    all_p_et_M = []
    all_r_et_M = []

    no_kw = 0
    all = 0

    for preds, true in zip(detected_keywords, gold_standard_keywords):
        preds = [x.split() for x in preds]

        true = [x.split() for x in true]

        metric_names = ['correct', 'precision', 'recall', 'f_score', 'precision_hard', 'f_score_hard']
        topk_range = [5, 10, 'k', 'M']

        match_list = get_match_result(true, preds)

        scores = run_metrics(match_list, preds, true, metric_names, topk_range)

        #print(scores)


        p_et_5 = scores['precision@5']
        r_et_5 = scores['recall@5']
        p_et_10 = scores['precision@10']
        r_et_10 = scores['recall@10']
        p_et_k = scores['precision@k']
        r_et_k = scores['recall@k']
        p_et_M = scores['precision@M']
        r_et_M = scores['recall@M']

        #print('Correct: ', scores['correct@10'], 'Len preds: ', len(preds), "p@10: ", p_et_10, 'hard_p@10: ', scores['precision_hard@10'])
        #print('Preds: ', preds)

        correct = []
        for w, m in zip(preds, match_list):
            if m > 0:
                correct.append(w)
        #print('Correct: ', correct)
        #print('Match : ', match_list)

        #print("P@10: ", p_et_10)
        #print("R@10: ", r_et_10)
        #print('---------------------------------------------------')

        all += 1

        if remove_docs_with_no_kw:
            if len(true) == 0:
                no_kw += 1
                continue

        all_p_et_5.append(p_et_5)
        all_r_et_5.append(r_et_5)
        all_p_et_10.append(p_et_10)
        all_r_et_10.append(r_et_10)
        all_p_et_k.append(p_et_k)
        all_r_et_k.append(r_et_k)
        all_p_et_M.append(p_et_M)
        all_r_et_M.append(r_et_M)

    #print()
    #print('Results: ')



    print('Num docs removed: ', no_kw)
    print('Num docs originally: ', all)
    print('Num docs evaluated: ', len(all_p_et_5))


    p_5, r_5, f_5 = macro_averaged_score(all_p_et_5, all_r_et_5)
    print("P@5: ", p_5)
    print("R@5: ", r_5)
    print("F1@5: ", f_5)
    print()


    p_10, r_10, f_10 = macro_averaged_score(all_p_et_10, all_r_et_10)
    print("P@10: ", p_10)
    print("R@10: ", r_10)
    print("F1@10: ", f_10)
    print()

    p_k, r_k, f_k = macro_averaged_score(all_p_et_k, all_r_et_k)
    print("P@k: ", p_k)
    print("R@k: ", r_k)
    print("F1@k: ", f_k)
    print()

    p_M, r_M, f_M = macro_averaged_score(all_p_et_M, all_r_et_M)
    print("P@M: ", p_M)
    print("R@M: ", r_M)
    print("F1@M: ", f_M)
    print('-----------------------------------------------------------------------')

    return p_5, r_5, f_5, p_10, r_10, f_10, p_k, r_k, f_k, p_M, r_M, f_M

