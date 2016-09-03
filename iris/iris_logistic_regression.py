'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import random
import csv

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

dataset = list(csv.reader(open('./iris.data')))
random.shuffle(dataset)

trainset = dataset[:80]
testset = dataset[80:]

def split_data(dataset):
    hot_encoding = {
        'Iris-setosa': [1., 0., 0.],
        'Iris-versicolor': [0., 1., 0.],
        'Iris-virginica': [0., 0., 1.]
    }
    x_data = [[float(x) for x in ins[:4]] for ins in dataset]
    y_data = [hot_encoding[ins[-1]] for ins in dataset]

    return x_data, y_data

def next_batch(batch_size=10):
    random.shuffle(trainset)
    return split_data(trainset[:batch_size])


# Parameters
learning_rate = 0.01
training_epochs = 150
batch_size = 10
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 4]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 3]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(trainset)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs, batch_ys = next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    x_test, y_test = split_data(testset)
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))
