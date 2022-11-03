

import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

ID = 'id'
TEXT = 'text'
SRC = 'source'

FOLD = 'fold'
PRED = 'prediction'
GT = 'ground_truth'
EP = 'epoch'

TP = 'true_positive'
FP = 'false_positive'
TN = 'true_negative'
FN = 'false_negative'

def object_eu(row):
    # post was not annotated
    if np.isnan(row['group']):
        return np.nan

    # post was annotated, but group was false
    if np.isnan(row['distinguishable_characteristic']):
        return False

    # post was fully annotated
    return (row['group'] or row['individual_as_group_member']) and row['distinguishable_characteristic']


def act_eu(row):
    return row['inciting_hatred'] or row['inciting_violence']


def scores_per_fold(predictions_df):
    names = ['fold', 'epoch', 'precision', 'recall', 'f1', 'accuracy', 'tn', 'fp', 'fn', 'tp']
    entries = []
    for i, fold in predictions_df.groupby(FOLD):
        results = []
        for e, predictions in fold.groupby(EP):
            pred = predictions[PRED].astype(bool).values
            gt = predictions[GT].astype(bool).values
            precision, recall, fscore, _ = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0.0)
            acc = accuracy_score(gt, pred)
            tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
            results.append([i, e, precision, recall, fscore, acc, tn, fp, fn, tp])
        entries.extend(results)
    df = pd.DataFrame(entries, columns=names).set_index(FOLD)
    return df

def scores(predictions_df):
    pred = predictions_df[PRED].astype(bool).values
    gt = predictions_df[GT].astype(bool).values
    precision, recall, fscore, _ = precision_recall_fscore_support(gt, pred, average='binary', zero_division=0.0)
    acc = accuracy_score(gt, pred)
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    return [precision, recall, fscore, acc, tn, fp, fn, tp]


def best_per_fold(metric_df, metric='f1'):
    best = []
    for i_fold, metrics in metric_df.groupby(FOLD):
        i_max = metrics[metric].argmax()
        best.append(metrics.iloc[i_max, :])
    best_df = pd.DataFrame(best)
    best_df.index.name = FOLD
    return best_df


if __name__ == '__main__':
    data_dir = 'data/training/logging'
    target_annos = [
        'punishable',
        'group',
        'distinguishable_characteristic',
        'individual_as_group_member',
        'inciting_hatred',
        'inciting_violence'
    ]

    posts = pd.read_csv('data/posts.csv').set_index(ID)[[SRC, TEXT]]

    score = []
    anno_preds = []
    for anno in target_annos:
        #read data
        anno_dir = os.path.join(data_dir, anno)
        out_dir = anno_dir

        test_predictions = pd.read_csv(os.path.join(anno_dir, 'test_results.csv')).astype({'id': int})
        train_predictions = pd.read_csv(os.path.join(anno_dir, 'train_results.csv')).astype({'id': int})

        test_scores = scores_per_fold(test_predictions)
        train_scores = scores_per_fold(train_predictions)

        test_scores.to_csv(os.path.join(anno_dir, 'test_scores.csv'))
        train_scores.to_csv(os.path.join(anno_dir, 'train_scores.csv'))

        # find best epoch for model of each fold
        best_test = best_per_fold(test_scores)
        best_train = best_per_fold(train_scores)

        best_test.to_csv(os.path.join(anno_dir, 'test_scores_best.csv'))
        best_train.to_csv(os.path.join(anno_dir, 'train_scores_best.csv'))

        train_predictions['split'] = 'train'
        test_predictions['split'] = 'test'
        train_scores['split'] = 'train'
        test_scores['split'] = 'test'

        # errors
        idx = [FOLD, EP]

        best_models = best_test.reset_index().set_index(idx).index
        preds = test_predictions.set_index([FOLD, EP]).loc[best_models, :]
        preds[TP] = (preds[PRED] == True) & (preds[GT] == True)
        preds[FP] = (preds[PRED] == True) & (preds[GT] == False)
        preds[TN] = (preds[PRED] == False) & (preds[GT] == False)
        preds[FN] = (preds[PRED] == False) & (preds[GT] == True)

        preds = preds.reset_index().set_index(ID).join(posts)

        # positive and negative cases
        POS = 'positive'
        NEG = 'negative'
        preds[POS] = preds[GT] == True
        preds[NEG] = preds[GT] == False

        preds.to_csv(os.path.join(out_dir, 'best_predictions.csv'))
        anno_preds.append(preds)

        # confusion matrices
        cols = [TP, FP, TN, FN, POS, NEG, SRC]
        per_source = preds[cols].groupby(SRC)
        confusion_per_source = per_source.sum()
        confusion_per_source_rel_by_label = confusion_per_source / per_source.count()

        confusion_per_source.to_csv(os.path.join(out_dir, 'confusion_per_source.csv'))
        confusion_per_source_rel_by_label.to_csv(os.path.join(out_dir, 'confusion_per_source_relative.csv'), float_format='%.3f')

        score.append(scores(preds))

    dataset = pd.read_csv('data/training/dataset.csv').drop(columns='text').astype(int).set_index('id')
    SUB = 'punishable_submodel'

    all_preds = pd.DataFrame({anno: pred[PRED] for anno, pred in zip(target_annos, anno_preds)})
    all_preds.loc[:, 'object'] = all_preds.apply(object_eu, axis=1)
    all_preds.loc[:, 'act'] = all_preds.apply(act_eu, axis=1)
    all_preds.loc[:, SUB] = np.min(all_preds[['object', 'act']], 1).astype('bool') # only true, if both true

    all_preds = all_preds[[SUB] + target_annos]
    all_preds.index = [int(i) for i in all_preds.index]

    all_preds = all_preds.astype(int).reset_index()\
        .join(dataset.reset_index(), rsuffix='_GT')\
        .sort_index(axis=1)\
        .set_index('id')\
        .join(posts)\
        .drop(columns='index')

    all_preds.to_csv(os.path.join(data_dir, 'predictions.csv'))

    # evaluate punishable for combination of submodels
    punishable_preds = anno_preds[0]
    submodel_preds = punishable_preds[[GT]].reset_index().join(all_preds[[SUB]].reset_index(), rsuffix='_r')\
        .drop(columns='id_r')\
        .set_index('id')
    submodel_preds[PRED] = submodel_preds[SUB]
    score.append(scores(submodel_preds))

    submodel_preds[TP] = (submodel_preds[PRED] == True) & (submodel_preds[GT] == True)
    submodel_preds[FP] = (submodel_preds[PRED] == True) & (submodel_preds[GT] == False)
    submodel_preds[TN] = (submodel_preds[PRED] == False) & (submodel_preds[GT] == False)
    submodel_preds[FN] = (submodel_preds[PRED] == False) & (submodel_preds[GT] == True)

    submodel_preds.index = [int(i) for i in submodel_preds.index]

    # store results
    submodel_path = os.path.join(data_dir, 'submodel')
    if not os.path.exists(submodel_path):
        os.makedirs(submodel_path)
    submodel_preds\
        .drop(columns=SUB)\
        .join(posts)\
        .to_csv(os.path.join(submodel_path, 'best_predictions.csv'))

    all_anno_names = target_annos + [SUB]
    metrics = ['precision', 'recall', 'f1', 'accuracy', 'tn', 'fp', 'fn', 'tp']
    score_df = pd.DataFrame(score, index=all_anno_names, columns=metrics)
    score_df.to_csv(os.path.join(data_dir, 'metrics.csv'))