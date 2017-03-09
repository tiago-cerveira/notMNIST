import numpy as np
import tensorflow as tf
from scipy import ndimage
import matplotlib.pyplot as plt
import pickle


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28  # change this using .shape
num_labels = 10  # change this using len(.unique)
num_channels = 1  # grayscale


batch_size = 1
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # Input data.
    pattern = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    layer3_weights = tf.Variable(tf.truncated_normal(
        [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    saver = tf.train.Saver()

    # Model.
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(pool + layer1_biases)

        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding='SAME')
        pool = tf.nn.max_pool(conv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(pool + layer2_biases)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        drop = tf.nn.dropout(hidden, 1)

        output_layer = tf.matmul(drop, layer4_weights) + layer4_biases
        return output_layer

    classification = tf.nn.softmax(model(pattern))


with tf.Session(graph=graph) as session:
    saver.restore(session, "/tmp/model.ckpt")
    print("Model restored.")


    def reformat(dataset):
        dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        return dataset
    print(session.run(layer1_biases))
    sample_images = reformat(test_dataset[2])

    dict = {pattern: sample_images}
    print(np.argmax(session.run(classification, feed_dict=dict)[0]))
    plt.imshow(test_dataset[2])
    plt.title("Some image")
    plt.show()



