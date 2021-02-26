import numpy as np
import csv

from constants import TEST_DATA_PATH, TRAIN_DATA_PATH


def load_data(fpath:str):
    X = []
    y = []
    with open(fpath,'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            X.append(line[1:-1])
            y.append(line[-1])
    X = np.array(X).astype(np.int64)
    y = np.array(y).astype(np.int)
    return X, y

def main():
    # Get data
    X_train, y_train = load_data(TRAIN_DATA_PATH)
    X_test, y_test = load_data(TEST_DATA_PATH)

    # get NN model


    # Train and save model
    

    # evaluate model


if __name__=='__main__':
    main()