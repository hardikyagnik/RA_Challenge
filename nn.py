import numpy as np

def sigmoid(Z):
    return 1/(1+ np.exp(-Z))
def delta_sigmoid(Z):
    return sigmoid(Z)*(1 - sigmoid(Z))

def get_weight_parameters(x_nodes,h1_nodes, h2_nodes, o_nodes):
    W1 = np.random.randn(h1_nodes, x_nodes)*0.01
    b1 = np.zeros((h1_nodes, 1))
    W2 = np.random.randn(h2_nodes, h1_nodes)*0.01
    b2 = np.zeros((h2_nodes, 1))
    W3 = np.random.randn(o_nodes, h2_nodes)*0.01
    b3 = np.zeros((o_nodes, 1))

    weight_parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3
        }
    return weight_parameters

def update_weight_parameters(weight_params, backward_vars, learning_rate):
    weight_params['W1'] = weight_params['W1'] - learning_rate*backward_vars['dW1']
    weight_params['b1'] = weight_params['b1'] - learning_rate*backward_vars['db1']
    weight_params['W2'] = weight_params['W2'] - learning_rate*backward_vars['dW2']
    weight_params['b2'] = weight_params['b2'] - learning_rate*backward_vars['db2']
    weight_params['W3'] = weight_params['W3'] - learning_rate*backward_vars['dW3']
    weight_params['b3'] = weight_params['b3'] - learning_rate*backward_vars['db3']
    return weight_params

def calculate_cost(Y_pred, Y_target): # cross entropy
    Y_pred = np.clip(Y_pred, 1e-15, 1-1e-15)
    return - Y_target*np.log(Y_pred) - (1 - Y_target)*np.log(1 - Y_pred)


def forward_propagation(weight_params, X):
    # ------- Forward Propogation -------
    # L1
    Z1 = np.dot(weight_params['W1'],X) + weight_params['b1']
    A1 = None
    A1 = sigmoid(Z1)

    # L2
    Z2 = np.dot(weight_params['W2'],A1) + weight_params['b2']
    A2 = sigmoid(Z2)

    # L3
    Z3 = np.dot(weight_params['W3'],A2) + weight_params['b3']
    A3 = sigmoid(Z3)

    forward_vars = {
        'Z1':Z1,
        'A1':A1,
        'Z2':Z2,
        'A2':A2,
        'Z3': Z3,
        'A3': A3
    }
    return forward_vars

def backward_propagation(weight_params, forward_vars, X, Y, activation):
    # ------- Backward Propogation -------
    # L3
    m = X.shape[1]
    dL3 = (-1/m) * np.multiply(Y/forward_vars['A3'] - (1-Y)/(1-forward_vars['A3']), delta_sigmoid(forward_vars['Z3']))
    dW3 = np.dot(dL3, forward_vars['A2'].T)
    db3 = np.sum(dL3, axis = 1)

    # L2
    dL2 = (delta_sigmoid(forward_vars['Z2']) * np.dot(weight_params['W3'].T, dL3))
    dW2 = np.dot(dL2, forward_vars['A1'].T)
    db2 = np.sum(dL2, axis = 1)

    # L1
    dL1 = (delta_sigmoid(forward_vars['Z1']) * np.dot(weight_params['W2'].T, dL2))
    dW1 = np.dot(dL1, X.T)
    db1 = np.sum(dL1, axis = 1)

    backward_vars = {
        'dW1':dW1,
        'db1':db1.reshape(weight_params['b1'].shape),
        'dW2':dW2,
        'db2':db2.reshape(weight_params['b2'].shape),
        'dW3':dW3,
        'db3':db3.reshape(weight_params['b3'].shape),
    }
    return backward_vars
