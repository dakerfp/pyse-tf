
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

