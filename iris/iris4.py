
import csv
import random
import tensorflow as tf

csv_reader = csv.reader(open('iris.data'))

def argmax(values):
    maxi = 0
    maxv = values[0]
    for i, v in enumerate(values):
        if v > maxv:
            maxv = maxv
            maxi = i
    return maxi

# hot encoding
def from_csv_row(row):
    attrs = [float(v) for v in row[:-1]]
    classes_map = {
        'Iris-virginica': [1, 0, 0],
        'Iris-setosa': [0, 1, 0],
        'Iris-versicolor': [0, 0, 1]
    }
    return (attrs, classes_map[row[-1]])

instances = [from_csv_row(row) for row in csv_reader]
random.shuffle(instances)

pivot = int(len(instances) * 0.9)
trainset = instances[:pivot]
testset = instances[pivot:]

x = tf.placeholder(tf.float32, [1, 4])
y_ = tf.placeholder(tf.float32, [1, 3])

W = tf.Variable(tf.random_uniform([3, 4]))
b = tf.Variable(tf.random_uniform([3, 1]))
y = tf.sigmoid(tf.matmul(W, tf.transpose(x)) + b)

mse = tf.nn.softmax_cross_entropy_with_logits(tf.transpose(y), y_)
loss = tf.reduce_mean(mse)

optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for i in range(120): # epochs
        random.shuffle(trainset)
        for attrs, cls in trainset:
            _, err = sess.run([train, loss], feed_dict={x: [attrs], y_: [cls]})

        total = len(testset)
        acc = 0
        mseacc = 0
        for attrs, cls in testset:
            result = sess.run([y], feed_dict={x: [attrs], y_: [cls]})
            # print(result)
            if argmax(result[0]) == argmax(cls):
                acc += 1

        print("Iteration ", i, "Accuracy ", float(acc) / total)

