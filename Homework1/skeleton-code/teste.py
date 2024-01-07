from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

'''
Receives:
    inputs: observations array of shape (n x (p+1)) accounting for bias
    labels: array of target values of shape (n, )
    W: initial array of weights of shape (n_classes x (p+1) )
    eta: scalar for learning rate
Returns:
    w: updated weights
'''
def multi_class_lr_epoch(inputs, labels, W, eta):
    # For each observation in data
    for x, y in zip(inputs, labels):
        i = W.dot(x)
        print(f"i: {i}")
        
        # Get probability scores according to the model (num_labels x 1).
        label_scores = np.expand_dims(W.dot(x), axis = 1)

        # One-hot encode true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(W, 0),1))
        y_one_hot[y] = 1

        # Softmax function
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        
        # SGD update. W is num_labels x num_features.
        W = W + eta * (y_one_hot - label_probabilities).dot(np.expand_dims(x, axis = 1).T)
    return W

data = load_digits()

inputs = data.data  # num_examples x num_features
labels = data.target  # num_examples x num_labels

n, p = np.shape(inputs)
n_classes = len(np.unique(labels))  # labels are 0, 1, ..., num_labels-1

print(f'There are {n} observations with {p} features classified into {n_classes} classes.')

# Augment points with a dimension for the bias.
inputs = np.concatenate([np.ones((n, 1)), inputs], axis=1)

# Observation example
plt.matshow(data.images[17])
plt.show()