
import csv
import tensorflow as tf

csv_reader = csv.reader(open('iris.data'))

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

pivot = int(len(instances) * 0.9)
trainset = instances[:pivot]
testset = instances[pivot:]

x = tf.placeholder(tf.float32, [1, 4])
y_ = tf.placeholder(tf.float32, [1, 3])

W = tf.Variable(tf.random_uniform([3, 4]))
b = tf.Variable(tf.random_uniform([3, 1]))
y = tf.matmul(W, tf.transpose(x)) + b

mse = tf.squared_difference(tf.transpose(y), y_)
loss = tf.reduce_mean(mse)
print(mse)
print(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    acc = 0
    for attrs, cls in testset:
        print(attrs, cls)
        result, err = sess.run([y, loss], feed_dict={x: [attrs], y_: [cls]})
