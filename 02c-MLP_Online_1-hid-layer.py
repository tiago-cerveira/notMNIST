import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    #del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28  # change this using .shape
num_labels = 10  # change this using len(.unique)


def reformat(dataset, labels):
    """
    data as a flat matrix
    labels as 1-hot encoding
    """
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 128  # 128
hidden_units = 1024
beta = 0.005

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    not_drop_prob = tf.constant(0.5)

    # Variables.
    weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units]))
    biases_1 = tf.Variable(tf.zeros([hidden_units]))

    weights_2 = tf.Variable(tf.truncated_normal([hidden_units, num_labels]))
    biases_2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    def propagate_net(data):
        hidden_layer = tf.nn.relu(tf.matmul(data, weights_1) + biases_1)
        drop = tf.nn.dropout(hidden_layer, not_drop_prob)
        output_layer = tf.matmul(drop, weights_2) + biases_2
        return output_layer

    output_layer = propagate_net(tf_train_dataset)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=output_layer))
    loss += beta * (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2))

    # Optimizer
    global_step = tf.Variable(0)
    learn_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.9)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(output_layer)
    valid_prediction = tf.nn.softmax(propagate_net(tf_valid_dataset))
    test_prediction = tf.nn.softmax(propagate_net(tf_test_dataset))

num_steps = 10000


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%\n" % accuracy(valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))


    def label(ind):
        return 'ABCDEFGHIJ'[ind.item(0)]

    for i in range(5):
        rand_val = np.random.randint(len(test_dataset))
        pred_class = label(np.argmax(test_prediction.eval()[rand_val]))
        actual_class = label(np.argmax(test_labels[rand_val]))

        plt.imshow(save["test_dataset"][rand_val])
        plt.title("Predicted, (actual): {}, ({})".format(pred_class, actual_class))
        plt.show()
