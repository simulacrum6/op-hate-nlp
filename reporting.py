import os
import csv
from datetime import datetime


def write_csv(file, values, sep=',', mode='w', quotechar='|', quoting=csv.QUOTE_MINIMAL):
    with open(file, mode, newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=sep, quotechar=quotechar, quoting=quoting)
        for entry in values:
            writer.writerow(entry)


class Logger:
    def __init__(self, log_dir, subdir, results_header, test_file_name='test_results.csv', train_file_name='train_results.csv'):
        self.log_dir = os.path.join(log_dir, subdir)
        self.subdir = subdir
        self.results_header = results_header
        self.test_records = os.path.join(self.log_dir, test_file_name)
        self.train_records = os.path.join(self.log_dir, train_file_name)
        self.train_start_time = datetime.now()
        self.epoch_start_time = datetime.now()

        self.create_logfiles()

    def create_logfiles(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        write_csv(self.test_records, [self.results_header])
        write_csv(self.train_records, [self.results_header])

    def append_results(self, records, train=True):
        file = self.train_records if train else self.test_records
        write_csv(file, records, mode='a')

    def log(self, message, *args, **kwargs):
        print(message.format(*args, **kwargs))

    def log_cv_training_start(self, n_folds, n_epochs, batch_size):
        time = datetime.now()
        self.train_start_time = time
        self.log(f'[{time}]: Starting Training for task "{self.subdir}", using cv. folds: {n_folds}, batch_size: {batch_size}, epochs: {n_epochs}')


    def log_cv_training_end(self):
        time = datetime.now()
        cv_train_time = (time - self.train_start_time).total_seconds()
        print(f'[{datetime.now()}]: done!'
              f'Training took {cv_train_time:.2f}seconds ({cv_train_time / 3600:.2f} hours)')

    def log_epoch_training_start(self, i_fold, i_epoch, n_folds, n_epochs):
        time = datetime.now()
        self.epoch_start_time = time
        percent = 100 * (i_fold + 1) * (i_epoch + 1) / (n_folds * n_epochs)
        self.log(f'[{time}]: training fold {i_fold + 1}/{n_folds} at epoch {i_epoch + 1}/{n_epochs} ({percent:.1f}%)')

    def log_epoch_training_end(self):
        time = datetime.now()
        train_time = (time - self.epoch_start_time).total_seconds()
        self.log(f'[{datetime.now()}]: done!'
              f'Training epoch took {train_time}seconds ({train_time / 3600:.2f} hours)')

    def log_eval(self, measure, measure_name, validation=True):
        split = 'Validation' if validation else 'Training'
        self.log(f'{split} Performance: {measure_name}={measure}')