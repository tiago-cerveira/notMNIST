from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time


def get_input():
    parser = argparse.ArgumentParser(description='Input info')
    parser.add_argument("model_size", type=int)

    return parser.parse_args()

start = time.time()
arg = get_input()

# load the dataset file (dictionary)
datasets = pickle.load(open('notMNIST.pickle', 'rb'))

train_sample = datasets['train_dataset'][:arg.model_size,:,:]
train_sample_labels = datasets['train_labels'][:arg.model_size]

(samples, width, height) = train_sample.shape
train_sample = np.reshape(train_sample, (samples, width * height))

(samples, width, height) = datasets['test_dataset'].shape
test_dataset = np.reshape(datasets['test_dataset'], (samples, width * height))

model = LogisticRegression(penalty='l2', C=1.0, n_jobs=-1)
model.fit(train_sample, train_sample_labels)

train_score = model.score(train_sample, train_sample_labels)
test_score = model.score(test_dataset, datasets['test_labels'])
print('Training score =', train_score)
print('Test score =', test_score)
print("took", round(time.time() - start, 3), "seconds")

amount = int(input("How many patterns at random would you like the system to predict? (0 to exit)"
                   "\nA: "))
print("Let's predict", amount, "patterns at random! :)")


def label(ind):
    """
    Convert class from an integer to a single letter
    :param ind: integer that relates to a class
    :return: class as a character
    """
    return 'ABCDEFGHIJ'[ind.item(0)]


predictions = model.predict(test_dataset)

for i in range(amount):
    rand_val = np.random.randint(len(datasets["test_dataset"]))
    pred_class = label(predictions[rand_val])
    actual_class = label(datasets["test_labels"][rand_val])
    plt.imshow(datasets["test_dataset"][rand_val])
    plt.title("Predicted, (actual): {}, ({})".format(pred_class, actual_class))
    plt.show()
