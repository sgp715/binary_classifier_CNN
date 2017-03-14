import utils
import tensorflow as tf
import numpy as np
import sys
import os


def conv_layer(X, category, kernel, strides, number):
    """
    inputs: dimensions of convolutional layers and pooling
    output: the pooling layer
    """
    output_size = kernel[3]
    with tf.variable_scope(category + '/') as scope:
        kernel = tf.Variable(tf.truncated_normal(kernel, dtype=tf.float32,
                                               stddev=1e-1), name='conv/weights' + str(number))
        conv = tf.nn.conv2d(X, kernel, strides, padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[output_size], dtype=tf.float32),
                           trainable=True, name='conv/biases' + str(number))
        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, 'conv/relu' + str(number))

    return conv


def pool_layer(X, category, ksize, kstrides, number):
    """
    inputs: takes in the parameters to create a pooling layer
    output: pooling tensor
    """

    with tf.variable_scope(category + '/') as scope:
        pool = tf.nn.max_pool(X,
                              ksize=ksize,
                              strides=kstrides,
                              padding='SAME',
                              name='pool' + str(number))

    return pool


def full_layer(X, category, input_size, output_size, number):
    """
    inputs: dimensions of fully connected layer
    output: the fully connected layer
    """

    # with tf.name_scope('full' + str(number)) as scope:
    with tf.variable_scope(category + '/') as scope:
        W = tf.Variable(tf.random_normal([input_size, output_size]), 'full/weights' + str(number))
        b = tf.Variable(tf.constant([output_size], dtype=tf.float32), 'full/biases' + str(number))
        full = tf.matmul(X, W) + b

    return full


def model(X, category, p_hidden):
    """
    output: return the model
    """

    conv1 = conv_layer(X, category, [5, 5, 3, 64], [1, 1, 1, 1], 1)
    pool1 = pool_layer(conv1, category, [1, 2, 2, 1], [1, 2, 2, 1], 1)

    conv2 = conv_layer(pool1, category, [5, 5, 64, 128], [1, 1, 1, 1], 2)
    pool2 = pool_layer(conv2, category, [1, 2, 2, 1], [1, 2, 2, 1], 2)

    conv3 = conv_layer(pool2, category, [3, 3, 128, 256], [1, 1, 1, 1], 3)
    pool3 = pool_layer(conv3, category, [1, 2, 2, 1], [1, 2, 2, 1], 3)

    reshape_pool3 = tf.reshape(pool3, [-1,  8*8*256])

    full1 = full_layer(reshape_pool3, category, 8*8*256, 4096, 1)

    #TODO: dropout having naming issue
    # full1 = tf.nn.dropout(full1, category, p_hidden)

    model = full_layer(full1, category, 4096, 2, 2)

    return model, conv1, conv2, conv3


class Net(object):

    @staticmethod
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

    def __init__(self, directory, category):

        model_path = directory + '/' + category
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.category_graph = tf.Graph()

        with self.category_graph.as_default():

            self.sess = tf.Session(graph=self.category_graph)

            self.model_location = model_path + '/' + 'model.ckpt'
            model_meta_path = self.model_location + '.meta'

            self.X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='X')
            self.Y = tf.placeholder(tf.float32, shape=[None, 2], name='Y')
            self.p_hidden = tf.placeholder(tf.float32, name='p_hidden')
            self.logits, self.c1, self.c2, self.c3 = model(self.X, category, self.p_hidden)
            self.saver = tf.train.Saver()

            self.saved_model = False
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt:
                self.saver = tf.train.import_meta_graph(model_meta_path)
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

                self.saved_model = True

    def train(self, data, labels, validation_data, validation_labels):
        """
        train model
        """

        with self.category_graph.as_default():

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
            learning_rate = 0.0001

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

            self.sess.run(tf.global_variables_initializer())


            batch_size = 100
            epochs = 100
            print "Accuracy: "

            try:
                for epoch in range(epochs):
                    idxs = np.random.permutation(len(data))
                    batches = len(data) // batch_size
                    for batch in range(batches):
                        idx = idxs[batch * batch_size: (batch + 1) * batch_size]
                        self.sess.run(optimizer, feed_dict={self.X:data[idx], self.Y:labels[idx], self.p_hidden: 0.5})

                    # if epoch % 5 == 0:
                    output = np.array(self.sess.run(self.logits, feed_dict={self.X: validation_data, self.p_hidden: 1.0}))
                    accuracy = self.compute_accuracy(output, validation_labels)
                    print "epoch "+ str(epoch) + ": " + str(accuracy) + '%'

            except KeyboardInterrupt:
                print "Saving model before exiting"
                self.saver.save(self.sess, self.model_location)
                self.saved_model = True
                exit()

            print "Saving model"
            self.saver.save(self.sess, self.model_location)

    def test(self, data, labels):
        """
        input: data and labels of test data
        prints out the accuracy on the test data
        """

        with self.category_graph.as_default():

            if self.saved_model:
                print "Testing..."
                output = self.sess.run(self.logits, feed_dict={self.X:data, self.Y:labels, self.p_hidden: 1.0})
                accuracy = self.compute_accuracy(output, labels)
                print "Accuracy: " + str(accuracy) + '%'
            else:
                print "Model does not exist yet...train first"

    def classify(self, image_path):
        """
        input: path to an image
        output: the classification of that image
        """

        with self.category_graph.as_default():

            if self.saved_model:
                img = [utils.load_image(image_path)]
                print image_path
                print "IMAGE"
                print img
                print "classifying..."
                output = np.argmax(self.sess.run(self.logits, feed_dict={self.X: img, self.p_hidden: 1.0})[0]).item()
		conv = self.sess.run([self.c1, self.c2, self.c3], feed_dict={self.X: img, self.p_hidden: 1.0})
                for idx,layer in enumerate(conv):
                    np.save('layer'+str(idx)+'.npy', layer)
                return output
            else:
                print "Model does not exist yet...train first"

    def __del__(self):

        with self.category_graph.as_default():

            self.sess.close()


if __name__ == "__main__":

    args = sys.argv[1:]
    def usage_message():
        print "usage:"
        print "python bcCNN.py -train <path/to/images1> <path/to/images2> <dir/to/save/model> <model/category>"
        print "python bcCNN.py -test <path/to/images1> <path/to/images2>"
        print "python bcCNN.py -classify <path/image/to/classify>"
        exit()

    if len(args) == 2:
	net = Net('.', 'formal')
        if args[0] == "-classify":
            path = args[1]
            print "classification: " + str(net.classify(path))
            exit()


    if len(args) == 5:

        label_1_images = args[1]
        if label_1_images[-1] == '/':
            label_1_images = label_1_images[:-1]
        label_0_images = args[2]
        if label_0_images[-1] == '/':
            label_0_images = label_0_images[:-1]
        directory = args[3]
        category = args[4]

        #instantiate object Net
        net = Net(directory, category)

        if args[0] == "-train":
            data, val_data, labels, val_labels = utils.get_test_and_validation_data(label_1_images, label_0_images)
            print "Initializing training..."
            net.train(data, labels, val_data, val_labels)
            exit()

        if args[0] == "-test":
            	print "here"
            	data, labels = utils.get_data(label_1_images, label_0_images)
            	net.test(data, labels)
            	exit()

    usage_message()
