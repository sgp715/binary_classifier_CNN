import utils
import tensorflow as tf
import numpy as np
import sys
import os


def conv_layer(X, kernel, strides, number):
    """
    inputs: dimensions of convolutional layers and pooling
    output: the pooling layer
    """
    output_size = kernel[3]

    with tf.name_scope('conv' + str(number)) as scope:
        kernel = tf.Variable(tf.truncated_normal(kernel, dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(X, kernel, strides, padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[output_size], dtype=tf.float32),
                           trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope)

    return conv

def pool_layer(X, ksize, kstrides, number):
    """
    inputs: takes in the parameters to create a pooling layer
    output: pooling tensor
    """

    pool = tf.nn.max_pool(X,
                           ksize=ksize,
                           strides=kstrides,
                           padding='VALID',
                           name='pool' + str(number))

    return pool

def full_layer(X, input_size, output_size, number):
    """
    inputs: dimensions of fully connected layer
    output: the fully connected layer
    """

    with tf.name_scope("full" + str(number)) as scope:
        W = tf.Variable(tf.random_normal([input_size, output_size]), name='W')
        b = tf.Variable(tf.constant([output_size], dtype=tf.float32), name='b')
        full = tf.matmul(X, W) + b

    return full

def model(X):
    """
    output: return the model
    """

    conv1 = conv_layer(X, [11, 11, 3, 64], [1, 4, 4, 1], 1)
    pool1 = pool_layer(conv1, [1, 3, 3, 1], [1, 2, 2, 1], 1)
    conv2 = conv_layer(pool1, [5, 5, 64, 128], [1, 1, 1, 1], 2)
    pool2 = pool_layer(conv2, [1, 3, 3, 1], [1, 2, 2, 1], 2)
    conv3 = conv_layer(pool2, [3, 3, 128, 256], [1, 1, 1, 1], 3)
    pool3 = pool_layer(conv3, [1, 3, 3, 1], [1, 2, 2, 1], 3)

    reshape_pool3 = tf.reshape(pool3, [-1, 256])

    full1 = full_layer(reshape_pool3, 256, 64, 1)
    #full2 = full_layer(full1, 64, 32, 2)
    model = full_layer(full1, 64, 2, 3)

    return model

X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')


def compute_accuracy(expected, actual):
    """
    input: the expected and actual
    output: the percent accuracy
    """

    assert(len(expected) == len(actual))

    length_data = len(expected)

    compare = np.array(np.argmax(expected, axis=1) == np.argmax(actual, axis=1))
    correct = (compare == True).sum()

    return (float(correct) / float(length_data)) * 100

def train(data, labels, validation_data, validation_labels):
    """
    train model
    """

    logits = model(X)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    saver = tf.train.Saver()

    batch_size = 50
    epochs = 100
    print "Accuracy: "
    with tf.Session() as sess:

        if os.path.isfile('model.ckpt.meta'):
            saver = tf.train.import_meta_graph('model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
        else:
            sess.run(tf.global_variables_initializer())

        try:
            for epoch in range(epochs):
                idxs = np.random.permutation(len(data))
                batches = len(data) // batch_size
                for batch in range(batches):
                    idx = idxs[batch * batch_size: (batch + 1) * batch_size]
                    sess.run(optimizer, feed_dict={X:data[idx], Y:labels[idx]})

                if epoch % 20 == 0:
                    output = np.array(sess.run(logits, feed_dict={X: validation_data}))
                    accuracy = compute_accuracy(output, validation_labels)
                    print accuracy + '%'
        except KeyboardInterrupt:
            print "Saving model before exiting"
            saver.save(sess, "model.ckpt")
            exit()

    print "Saving model"
    saver.save(sess, "model.ckpt")


def test(data, labels):
    """
    input: data and labels of test data
    prints out the accuracy on the test data
    """

    logits = model(X)
    saver = tf.train.Saver()

    with tf.Session() as sess:

        if os.path.isfile('model.ckpt.meta'):
            saver = tf.train.import_meta_graph('model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            output = sess.run(logits, feed_dict={X:data, Y:labels})
            accuracy = compute_accuracy(output, labels)
            print accuracy
        else:
            print "Model does not exist yet...train first"

def classify(image_path):
    """
    input: path to an image
    output: the classification of that image
    """

    img = [utils.load_image(image_path)]

    logits = model(X)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if os.path.isfile('model.ckpt.meta'):
            saver = tf.train.import_meta_graph('model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))
            output = np.argmax(sess.run(logits, feed_dict={X: img})[0])
            print output
            return output
        else:
            print "Model does not exist yet...train first"

if __name__ == "__main__":

    args = sys.argv[1:]
    def usage_message():
        print "usage:"
        print "net.py -train <path/to/data1> <path/to/data2>"
        print "net.py -test <path/to/data1> <path/to/data2>"
        print "net.py -classify <path/image/to/classify>"
        exit()

    if len(args) == 2:
        if args[0] == "-classify":
            path = args[1]
            classify(path)
            exit()

    if len(args) == 3:
        label_1_images = args[1]
        label_0_images = args[2]

        if args[0] == "-train":
            data, val_data, labels, val_labels = utils.get_test_and_validation_data(label_1_images, label_0_images)
            print "Initializing training..."
            train(data, labels, val_data, val_labels)
            exit()
        if args[0] == "-test":
            data, labels = utils.get_data(label_1_images, label_0_images)
            test(data, labels)
            exit()

    usage_message()
