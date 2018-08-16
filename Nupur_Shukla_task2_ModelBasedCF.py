from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import csv
import sys
import time
import random
import math

def category(diff):
	if(diff>=0 and diff<1):
		return 0
	elif(diff>=1 and diff<2):
		return 1
	elif(diff>=2 and diff<3):
		return 2
	elif(diff>=3 and diff<4):
		return 3
	elif(diff>=4):
		return 4

start = time.time()
ratingsFile = sys.argv[1]
testFile = sys.argv[2]

sc = SparkContext()
ratingsRdd = sc.textFile(ratingsFile, minPartitions=None, use_unicode=False)
ratingsRdd = ratingsRdd.mapPartitions(lambda x : csv.reader(x))
header1 = ratingsRdd.first()
ratingsRdd = ratingsRdd.filter(lambda x : x != header1)
ratingsRdd = ratingsRdd.map(lambda x : ((int(x[0]), int(x[1])), float(x[2])))

testRdd = sc.textFile(testFile, minPartitions=None, use_unicode=False)
testRdd = testRdd.mapPartitions(lambda x : csv.reader(x))
header2 = testRdd.first()
testRdd = testRdd.filter(lambda x : x != header2)
testRdd = testRdd.map(lambda x : ((int(x[0]), int(x[1])), 1))

trainingRdd = ratingsRdd.subtractByKey(testRdd)
trainingRdd = trainingRdd.map(lambda x : Rating(x[0][0], x[0][1], x[1]))
testRdd = testRdd.map(lambda x : (x[0][0], x[0][1]))

random.seed(10)
lambda_ = 0.1 # IMPORTANT : Change lambda_ parameter here for ml-20m file
model = ALS.train(trainingRdd, rank = 10, iterations = 7, lambda_ = lambda_)
predictions = model.predictAll(testRdd).map(lambda r: ((r[0], r[1]), r[2]))

outputFile = "Nupur_Shukla_task2_ModelBasedCF.txt"
outFile = open(outputFile, "w")
predictionsList = sorted(predictions.collect())
for prediction in predictionsList:
	outFile.write(str(prediction[0][0]) + ", " + str(prediction[0][1]) + ", " + str(prediction[1]) + "\n")
outFile.close()

ratesAndPreds = ratingsRdd.join(predictions)

differences = ratesAndPreds.map(lambda x: abs(x[1][0]-x[1][1])).collect()
diff = sc.parallelize(differences)
categories = diff.map(lambda x : (category(x), 1)).reduceByKey(lambda x, y : x+y).sortByKey().collect()
rmse = math.sqrt(diff.map(lambda x: x*x).mean())

print ">=0 and <1: ",categories[0][1]
print ">=1 and <2: ",categories[1][1]
print ">=2 and <3: ",categories[2][1]
print ">=3 and <4: ",categories[3][1]
print ">=4: ",categories[4][1]
print "RMSE: ", str(rmse)

end = time.time()
print "Time: ", end - start, " sec"