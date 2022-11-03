from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
from torch.nn import NLLLoss
from torch.nn.functional import log_softmax, softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from reporting import Logger
import sys


def lr_schedule(epoch):
    """Ramps up linearly for the first 10% of training and ramps down for the remainder."""
    peak = E * 0.1
    if epoch <= peak:
        return epoch / peak
    else:
        return 1 - ((epoch - peak) / (E - peak))


def to_tensor_dataset(data_frame, idxs):
    """Filters data according to given indices and returns it as TensorDataset for BERT Model.
    The resulting dataset contains 4 tensors: token ids, attention masks, ground truth labels and post ids.
    """
    sentences = data_frame.iloc[idxs, sent_idx].to_list()
    labels = data_frame.iloc[idxs, trgt_col_idx].astype(int).to_list()
    indices = data_frame.iloc[idxs, :].index
    inputs = tokenizer(sentences, return_tensors='pt', max_length=128, padding='max_length', truncation=True)

    return TensorDataset(inputs['input_ids'],
                         inputs['attention_mask'],
                         torch.tensor(labels),
                         torch.tensor(indices))


def generate_fold_datasets(data_frame, train_test_indices):
    """Filters data according to given tuple of indices and returns a train set and test set for the BERT Model."""
    i_train, i_test = train_test_indices

    train_set = to_tensor_dataset(data_frame, i_train)
    test_set = to_tensor_dataset(data_frame, i_train)

    return train_set, test_set


def predict(model, tokens, mask, loss_fn):
    """Predicts label for given tokens and returns tuple of (prediction, ground_truth, confidence, loss).
    """
    pred = model(tokens, mask)[0]
    probs = softmax(pred, 1)
    confs, pred_labels = torch.max(probs, 1)
    loss = loss_fn(log_softmax(pred, 1), targets)
    return pred, probs, confs, pred_labels, loss


def package_results(confs, pred_labels, loss, idx, step, n, fold):
    """Packages results for current batch as array of records.
    Each record contains prediction, ground_truth, confidence, loss, (post) id, global step, cv fold, epoch.
    """
    return np.array([
        pred_labels.cpu().numpy(),
        targets.cpu().numpy(),
        confs.detach().cpu().numpy(),
        np.full(n, loss.detach().cpu().numpy()),
        idx.cpu().numpy(),
        np.full(n, step),
        np.full(n, fold),
        np.full(n, e)
    ]).T

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 'parse' args
    args = sys.argv[1:]
    train_dir = args[0] if len(args) > 0 else 'data/training'
    filename = args[1] if len(args) > 1 else 'dataset.csv'
    save_models = args[2] if len(args) > 2 else False

    # set up output directories
    log_dir = os.path.join(train_dir, 'logging')
    model_dir = os.path.join(train_dir, 'models')

    for d in (train_dir, log_dir, model_dir):
        if not os.path.exists(d):
            os.mkdir(d)

    # read dataset and extract targets
    df = pd.read_csv(os.path.join(train_dir, filename), index_col=0)
    sent, *target_annos = df.columns
    sent_idx = 0

    for trgt_col_idx, trgt in enumerate(target_annos):
        trgt_col_idx = trgt_col_idx + 1

        # set up training
        bert = 'deepset/gbert-base'
        tokenizer = AutoTokenizer.from_pretrained(bert)
        E = 20
        B = 16
        n_folds = 10
        LR = 2e-5
        L = NLLLoss()
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=4669)

        # set up logging
        record_names = ['prediction', 'ground_truth', 'confidence', 'loss', 'id', 'step', 'fold', 'epoch']
        logger = Logger(log_dir, trgt, record_names)
        logger.log_cv_training_start(n_folds, E, B)

        for fold, idxs in enumerate(cv.split(df[sent], df[trgt])):
            train_set, test_set = generate_fold_datasets(df, idxs)
            N = len(train_set)

            # set up model
            model = AutoModelForSequenceClassification.from_pretrained(bert).to(device)
            opt = Adam(model.parameters(), lr=LR)
            scheduler = LambdaLR(opt, lr_schedule)

            best_f1 = -1.0
            for e in range(E):
                logger.log_epoch_training_start(fold, e, n_folds, E)

                model.train()
                for i, data in enumerate(DataLoader(train_set, batch_size=B, shuffle=True)):
                    opt.zero_grad()
                    tokens, mask, targets, idx = (x.to(device) for x in data)
                    pred, probs, confs, pred_labels, loss = predict(model, tokens, mask, L)
                    loss.backward()
                    opt.step()
                    scheduler.step(e + i * B / N)

                    step = (e * N + i)
                    n = len(targets)
                    logger.append_results(package_results(confs, pred_labels, loss, idx, step, n, fold), train=True)
                logger.log_epoch_training_end()

                # eval after epoch
                model.eval()
                for batch in DataLoader(test_set, batch_size=B, shuffle=False):
                    tokens, mask, targets, idx = (x.to(device) for x in batch)
                    pred, probs, confs, pred_labels, loss = predict(model, tokens, mask, L)

                    n = len(targets)
                    logger.append_results(package_results(confs, pred_labels, loss, idx, step, n, fold), train=False)

                # store best model
                evaluating = [False, True]
                records = [logger.train_records, logger.test_records]
                for eval_flag, record in zip(evaluating, records):
                    tr = pd.read_csv(record).query(f'fold == {fold}').query(f'epoch == {e}')
                    prec, rec, f1, _ = precision_recall_fscore_support(tr['ground_truth'].astype(bool),
                                                                       tr['prediction'].astype(bool),
                                                                       average='binary', pos_label=True)
                    acc = accuracy_score(tr['ground_truth'].astype(bool), tr['prediction'].astype(bool))
                    
                    logger.log_eval(acc, 'accuracy', eval_flag)
                    logger.log_eval(f1, 'f1', eval_flag)
                    logger.log_eval(prec, 'precision', eval_flag)
                    logger.log_eval(rec, 'recall', eval_flag)

                if best_f1 < f1 and save_models:
                    best_f1 = f1
                    model_name = f'model_{trgt}__retrain__f{fold}_best.pth'
                    torch.save(model.state_dict(), os.path.join(model_dir, model_name))
        
        logger.log_cv_training_end()


