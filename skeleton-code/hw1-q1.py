#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            # x_i is a 785x1 vector
            # y_i is a scalar
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # x_i is a 785x1 vector
        # y_i is a scalar
        # self.W is a 4x785 matrix

        y_hat = self.predict(x_i)

        # Update all weights if prediction is wrong
        if y_i != y_hat:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # x_i is a 785x1 vector
        # y_i is a scalar
        # self.W is a 4x785 matrix

        # Get probability scores according to the model (num_labels x 1).
        label_scores = np.expand_dims(np.dot(self.W, x_i), axis = 1)

        # One-hot encode true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0),1))
        y_one_hot[y_i] = 1

        # Softmax function
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))

        # SGD update. W is num_labels x num_features.
        self.W += learning_rate * (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # First is input size, last is output size.
        mu, sigma = 0.1, 0.1 # mean and standard deviation

        # Initialize weights with correct shapes 
        W1 = np.random.normal(mu, sigma, (hidden_size, n_features))
        b1 = np.zeros(hidden_size)

        W2 = np.random.normal(mu, sigma, (n_classes, hidden_size))
        b2 = np.zeros(n_classes)

        self.W = [W1, W2]
        self.b = [b1, b2]

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        predicted_labels = []
        for x in X:
            # Compute forward pass and get the class with the highest probability
            output, _ = self.forward(x)
            y_hat = np.argmax(output)
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def forward(self, x):
        num_layers = len(self.W)
        # Activation function (relu)
        g = lambda x: np.maximum(0, x)
        hiddens = []
        # compute hidden layers
        for i in range(num_layers):
            h = x if i == 0 else hiddens[i-1]
            z = self.W[i].dot(h) + self.b[i]
            if i < num_layers-1:  # Assuming the output layer has no activation.
                hiddens.append(g(z))
        #compute output
        output = z

        hiddens = np.array(hiddens)

        return output, hiddens

    def compute_loss(self, output, y):
        # compute loss
        max_output = np.max(output)
        probs = np.exp(output - max_output) / np.sum(np.exp(output - max_output))
        loss = -y.dot(np.log(probs))
        return loss

    def backward(self, x, y, output, hiddens):
        num_layers = len(self.W)

        max_output = np.max(output)
        probs = np.exp(output - max_output) / np.sum(np.exp(output - max_output))
        grad_z = probs - y  
        
        grad_weights = []
        grad_biases = []
        
        # Backpropagate gradient computations
        for i in range(num_layers-1, -1, -1):
            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens[i-1]

            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)

            # Gradient of hidden layer below.
            grad_h = self.W[i].T.dot(grad_z)

            # Gradient of hidden layer below before activation.
            grad_z = grad_h * np.where(h < 0, 0, 1)   # Grad of loss wrt z3.

        # Making gradient vectors have the correct order
        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    def train_epoch(self, X, y, learning_rate=0.001):

        """
        Dont forget to return the loss of the epoch.
        """

        num_layers = len(self.W)
        total_loss = 0

        # For each observation and target
        i = 0
        for x, y in zip(X, y):
            # One-hot encode true label (num_labels x 1).
            y_one_hot = np.zeros((np.size(self.W[-1], 0),1))
            y_one_hot[y] = 1
            y_one_hot = y_one_hot.reshape(-1,)

            # Comoute forward pass
            output, hiddens = self.forward(x)

            if i == 0:
                print(output.shape)
                print(y_one_hot.shape)
                print(hiddens[0].shape)
                i += 1

            # Compute Loss and Update total loss
            loss = self.compute_loss(output, y_one_hot)
            total_loss += loss

            # Compute backpropagation
            grad_weights, grad_biases = self.backward(x, y_one_hot, output, hiddens)

            # Update weights
            for i in range(num_layers):
                self.W[i] -= learning_rate*grad_weights[i]
                self.b[i] -= learning_rate*grad_biases[i]
        return total_loss / len(X)

    # def train_epoch(self, X, y, learning_rate=0.001):
    #     total_loss = 0
    #     for x_i, y_i in zip(X, y):

    #         h0 = x_i
    #         y_one_hot = np.zeros(self.B[2].shape[0])
    #         y_one_hot[y_i] = 1

    #         z1 = self.W1.dot(h0) + self.B[1]
    #         h1 = np.maximum(0, z1)  # ReLU

    #         z2 = self.W2.dot(h1) + self.B[2]

    #         # Numerically stable softmax
    #         max_z2 = np.max(z2)
    #         p = np.exp(z2 - max_z2) / np.sum(np.exp(z2 - max_z2))

    #         # Loss
    #         loss = -y_one_hot.dot(np.log(p))
    #         total_loss += loss

    #         # Backpropagation
    #         grad_z2 = p - y_one_hot  # Grad of loss wrt p

    #         # Gradient of hidden parameters.
    #         grad_W2 = grad_z2[:, None].dot(h1[:, None].T)
    #         grad_b2 = grad_z2

    #         # Gradient of hidden layer below.
    #         grad_h1 = self.W2.T.dot(grad_z2)

    #         # Gradient of hidden layer below before activation.
    #         grad_z1 = np.where(z1 > 0, grad_h1, 0)

    #         # Gradient of hidden parameters.
    #         grad_W1 = grad_z1[:, None].dot(h0[:, None].T)
    #         grad_b1 = grad_z1

    #         # Update weights
    #         self.W[1] -= learning_rate*grad_W1
    #         self.B[1] -= learning_rate*grad_b1
    #         self.W[2] -= learning_rate*grad_W2
    #         self.B[2] -= learning_rate*grad_b2

    #     return total_loss/X.shape[0]


def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)

    train_X, train_y = data["train"]
    # print(f"train_X shape: {train_X.shape}")
    train_X, train_y = train_X[:5000], train_y[:5000]

    dev_X, dev_y = data["dev"]
    # print(f"dev_X shape: {dev_X.shape}")
    dev_X, dev_y = dev_X[:100], dev_y[:100]

    test_X, test_y = data["test"]
    # print(f"test_X shape: {test_X.shape}")

    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []

    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]

        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )

        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
