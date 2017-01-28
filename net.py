import utils
import tensorflow as tf
import numpy as np


def conv_layer(X, kernel, strides, ksize, kstrides, number):
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

    pool = tf.nn.max_pool(conv,
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


# create the network
X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')

pool1 = conv_layer(X, [11, 11, 3, 64], [1, 4, 4, 1], [1, 3, 3, 1], [1, 2, 2, 1], 1)
pool2 = conv_layer(pool1, [5, 5, 64, 128], [1, 1, 1, 1], [1, 3, 3, 1], [1, 2, 2, 1], 2)
pool3 = conv_layer(pool2, [3, 3, 128, 256], [1, 1, 1, 1], [1, 3, 3, 1], [1, 2, 2, 1], 3)

reshape_pool3 = tf.reshape(pool3, [-1, 256])

full1 = full_layer(reshape_pool3, 256, 64, 1)
full2 = full_layer(full1, 64, 32, 2)
full3 = full_layer(full2, 32, 2, 3)

Y_pred = tf.nn.softmax(full3)

# have the cost be the
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_pred, labels=Y))
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

data, labels = utils.get_data("formal", "casual")

batch_size = 50
epochs = 100
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        idxs = np.random.permutation(len(data))
        batches = len(data) // batch_size
        for batch in range(batches):
            idx = idxs[batch * batch_size: (batch + 1) * batch_size]
            sess.run(optimizer, feed_dict={X:data[idx], Y:labels[idx]})

        training_cost = sess.run(cost, feed_dict={X:data, Y:labels})
        print training_cost
