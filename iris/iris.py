
import csv
import random

csv_reader = csv.reader(open('iris.data'))
instances = list(csv_reader)

pivot = int(len(instances) * 0.9)
trainset = instances[:pivot]
testset = instances[pivot:]
