import numpy as np
import csv
import pickle
import os
import matplotlib.pyplot as plt

from constants import *
from nn import (
    forward_propagation, backward_propagation, get_weight_parameters,
    update_weight_parameters, calculate_cost)


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


def evaluate(weight_params, X, y):
    forward_vars = forward_propagation(weight_params, X.T)
    y_hat = forward_vars['A3']
    y_hat = y_hat > 0.5
    y_hat = y_hat.reshape(-1).tolist()
    y = y.tolist()
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y[i]==1 and y[i]==y_hat[i]:
           TP += 1
        if y_hat[i]==1 and y[i]!=y_hat[i]:
           FP += 1
        if y[i]==0 and y[i]==y_hat[i]:
           TN += 1
        if y_hat[i]==0 and y[i]!=y_hat[i]:
           FN += 1
    
    return TP, FP, TN, FN


def save_model_weights(weights):
    import time
    unique = int(time.time()*100)
    output_path = os.path.join(BINARIES_PATH, f"weights_{unique}.pkl")
    with open(output_path, "wb") as fout:
        pickle.dump(weights, fout, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    # Get data
    X_train, y_train = load_data(TRAIN_DATA_PATH)
    X_test, y_test = load_data(TEST_DATA_PATH)
    m = len(X_train)

    # get NN model
    weight_params = get_weight_parameters(x_nodes=14, h1_nodes=10, h2_nodes=5, o_nodes=1)
    
    # Train model
    cost_log = []
    for epoch in range(1,EPOCH+1):
        forward_vars = forward_propagation(weight_params, X_train.T)
        loss_vec = calculate_cost(forward_vars['A3'], y_train.T)
        cost_log.append((1/m)*np.sum(loss_vec))
        backward_vars = backward_propagation(weight_params, forward_vars, X_train.T, y_train.T, activation="sigmoid")
        weight_params = update_weight_parameters(weight_params, backward_vars, LR)
        print(f"epoch={epoch}, loss={cost_log[-1]}")
    
    # Save model
    save_model_weights(weight_params)

    # evaluate model
    TP, FP, TN, FN = evaluate(weight_params, X_test, y_test)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print(f"Test Dataset precision = {precision}, reacll = {recall}")

    TP, FP, TN, FN = evaluate(weight_params, X_train, y_train)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print(f"Train Dataset precision = {precision}, reacll = {recall}")

    plt.plot(list(range(EPOCH)),cost_log)
    plt.xlabel("Training Iterations",color="Green")
    plt.ylabel("Avg Cross Entropy Loss",color="Green")
    plt.title("Error vs Iteration Chart", color="Blue")
    plt.show()


if __name__=='__main__':
    main()