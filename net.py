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
    full2 = full_layer(full1, 64, 32, 2)
    model = full_layer(full2, 32, 2, 3)

    return model

def train(model, data, labels):
    """
    train model
    """

    X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')

    model = model(X)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    saver = tf.train.Saver()

    batch_size = 50
    epochs = 100
    length_data = len(data)
    print "Accuracy on training: "
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
                    training_cost = sess.run(cost, feed_dict={X:data, Y:labels})
                    #print "cost: " + str(training_cost)
                    output = np.array(sess.run(model, feed_dict={X: data}))
                    compare = np.array(np.argmax(output, axis=1) == np.argmax(labels, axis=1))
                    correct = (compare == True).sum()
                    print str((float(correct) / float(length_data)) * 100) + '%'
        except KeyboardInterrupt:
            print "Saving model before exiting"
            saver.save(sess, "model.ckpt")
            exit()

    print "Saving model"
    saver.save(sess, "model.ckpt")


if __name__ == "__main__":

    args = sys.argv[1:]
    def usage_message():
        print "usage: net.py -train [path/to/data1] [path/to/data2]"
        exit()

    if len(args) != 3:
        usage_message()

    if args[0] == "-train":

        label_1_images = args[1]
        label_0_images = args[2]
        data, labels = utils.get_data(label_1_images, label_0_images)

        print "Initializing training..."
        train(model, data, labels)
    else:
        usage_message()
