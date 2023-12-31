from cProfile import label
from itertools import count
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import argparse
from splitted_sbp_dbp_features import getFeatures_SBP_DBP

def softmax(x):
    result = np.zeros(x.shape)
    for i in range(x.shape[0]):
        row = np.array(x[i])
        diff = np.max(row)
        row = row - diff
        row = np.exp(row)
        s = np.sum(row)
        result[i] = row/s
    return result

def sigmoid(x):
    def sigmoid_helper(x):
        if x<0:
            return np.exp(x)/(1+np.exp(x))
        else:
            return 1/(1+np.exp((-1)*x))

    sigmoid_helper_vec = np.vectorize(sigmoid_helper)
    return sigmoid_helper_vec(x)

def get_initial_params(input_size, num_hidden, num_output):
    w1 = np.zeros((input_size, num_hidden))
    for i in range(input_size):
        for j in range(num_hidden):
            w1[i,j] = np.random.normal()
    w2 = np.zeros((num_hidden, num_output))
    for i in range(num_hidden):
        for j in range(num_output):
            w2[i,j] = np.random.normal()
    b1 = np.zeros(num_hidden)
    b2 = np.zeros(num_output)
    return{
        "W1": w1,
        "W2": w2,
        "b1": b1,
        "b2": b2
    }

def forward_prop(data, labels, params):
    W1T = np.transpose(params["W1"])
    W2T = np.transpose(params["W2"])
    b1 = params["b1"]
    b2 = params["b2"]

    z_1 = np.transpose(np.matmul(W1T, np.transpose(data))) + np.tile(b1, (data.shape[0], 1))
    activations = sigmoid(z_1)
    z = np.transpose(np.matmul(W2T, np.transpose(activations))) + np.tile(b2, (data.shape[0], 1))
    outputs = softmax(z)

    total_loss = 0

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if(outputs[i,j]>0):
                total_loss += (labels[i,j]*np.log(outputs[i,j]))
    avg_loss = -1*(total_loss/data.shape[0])
    return (activations, outputs, avg_loss)

def backward_prop(data, labels, params, forward_prop_func):
    W1 = params["W1"]
    W2 = params["W2"]
    b1 = params["b1"]
    b2 = params["b2"]

    (activations, outputs, avg_loss) = forward_prop_func(data, labels, params)
    grad_losses = np.zeros(labels.shape)
    total_grad_loss = np.zeros(labels.shape[1])

    for i in range(labels.shape[0]):
        grad_losses[i]= outputs[i] - labels[i]
        total_grad_loss += outputs[i] - labels[i]
    delta_2 = total_grad_loss/data.shape[0]
    dj_db_2 = delta_2
    dj_dw_2 = np.matmul(np.transpose(grad_losses), activations)/data.shape[0]
    sigmoid_prime = np.multiply(activations,(np.ones(activations.shape)-activations))
    delta_1 = np.multiply(np.matmul(grad_losses, np.transpose(W2)), sigmoid_prime)
    dj_dw_1 = np.matmul(np.transpose(delta_1), data)/data.shape[0]
    dj_db_1 = np.sum(delta_1, axis=0)/data.shape[0]

    dW1 = np.transpose(dj_dw_1)
    dW2 = np.transpose(dj_dw_2)
    db1 = dj_db_1
    db2 = dj_db_2
    return{
        "W1": dW1,
        "W2": dW2,
        "b1": db1,
        "b2": db2
    }

def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    W1 = params["W1"]
    W2 = params["W2"]
    b1 = params["b1"]
    b2 = params["b2"]

    (activations, outputs, avg_loss) = forward_prop_func(data, labels, params)
    grad_losses = np.zeros(labels.shape)
    total_grad_loss = np.zeros(labels.shape[1])

    for i in range(labels.shape[0]):
        grad_losses[i]= outputs[i] - labels[i]
        total_grad_loss += outputs[i] - labels[i]
    delta_2 = total_grad_loss/data.shape[0]
    dj_db_2 = delta_2
    dj_dw_2 = np.matmul(np.transpose(grad_losses), activations)/data.shape[0]
    sigmoid_prime = np.multiply(activations,(np.ones(activations.shape)-activations))
    delta_1 = np.multiply(np.matmul(grad_losses, np.transpose(W2)), sigmoid_prime)
    dj_dw_1 = np.matmul(np.transpose(delta_1), data)/data.shape[0]
    dj_db_1 = np.sum(delta_1, axis=0)/data.shape[0]

    dW1 = np.transpose(dj_dw_1) + 2*reg*W1
    dW2 = np.transpose(dj_dw_2) + 2*reg*W2
    db1 = dj_db_1
    db2 = dj_db_2
    return{
        "W1": dW1,
        "W2": dW2,
        "b1": db1,
        "b2": db2
    }

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    start = 0
    while(start<train_data.shape[0]):
        end = start + batch_size
        derivatives = backward_prop_func(train_data[start:end, :], train_labels[start:end, :], params, forward_prop_func)
        params["W1"] -= learning_rate * derivatives["W1"]
        params["W2"] -= learning_rate * derivatives["W2"]
        params["b1"] -= learning_rate * derivatives["b1"]
        params["b2"] -= learning_rate * derivatives["b2"]
        start = end
    print("epoch")
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):

    (nexp, dim) = train_data.shape
    params = get_initial_params_func(dim, num_hidden, 5)
    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []

    for epoch in range(num_epochs):
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))
    
    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 5))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)
    np.save(name+"_params.npy", params)
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()
        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        fig.savefig('./' + name + '.png')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    return accuracy

def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()
    data = getFeatures_SBP_DBP()
    features = data["features"]
    labels = data["one-hot-labels"]   

    np.random.seed(100)
    train_data, train_labels = features[0:6000,:], labels[0:6000,:]

    dev_data = train_data[0:2000,:]
    dev_labels = train_labels[0:2000,:]
    train_data = train_data[2000:,:]
    train_labels = train_labels[2000:,:]
    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = features[6000:6500,:], labels[6000:6500,:]
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    
    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels, 
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)
        
    return baseline_acc, reg_acc

if __name__ == '__main__':
    main()