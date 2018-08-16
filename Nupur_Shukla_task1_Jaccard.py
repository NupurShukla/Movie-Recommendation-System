from pyspark import SparkContext
import csv
import sys
import time

def jaccardSimilarity(movie1, movie2):
	a = dataDict[movie1]
	b = dataDict[movie2]

	intersection = len(a & b)
	union = len(a) + len(b) - intersection

	return float(intersection)/float(union)

def bandComparison(signature1, signature2, B, R):
	i = 0
	while i<=(B-1):
		a = signature1[i:i+R]
		b = signature2[i:i+R]
		if a == b:
			return True
		i = i + R

	return False

def append(a, b):
	a.append(b)
	return a

def getSignatureMatrix():

	# Minhashing
	h1 = data.map(lambda x : (x[1], (7*x[0]+1) % 671)).groupByKey().mapValues(list)
	h1 = h1.map(lambda x : (x[0], min(x[1])))

	h2 = data.map(lambda x : (x[1], (3*x[0]+1) % 671)).groupByKey().mapValues(list)
	h2 = h2.map(lambda x : (x[0], min(x[1])))

	h3 = data.map(lambda x : (x[1], (51*x[0]+58) % 671)).groupByKey().mapValues(list)
	h3 = h3.map(lambda x : (x[0], min(x[1])))

	h4 = data.map(lambda x : (x[1], (68*x[0]+73) % 671)).groupByKey().mapValues(list)
	h4 = h4.map(lambda x : (x[0], min(x[1])))

	h5 = data.map(lambda x : (x[1], 5*(17*x[0]+19) % 671)).groupByKey().mapValues(list)
	h5 = h5.map(lambda x : (x[0], min(x[1])))

	h6 = data.map(lambda x : (x[1], (42*x[0]+78) % 671)).groupByKey().mapValues(list)
	h6 = h6.map(lambda x : (x[0], min(x[1])))

	h7 = data.map(lambda x : (x[1], 7*(17*x[0]+19) % 671)).groupByKey().mapValues(list)
	h7 = h7.map(lambda x : (x[0], min(x[1])))

	h8 = data.map(lambda x : (x[1], (136*x[0]+152) % 671)).groupByKey().mapValues(list)
	h8 = h8.map(lambda x : (x[0], min(x[1])))

	h9 = data.map(lambda x : (x[1], 9*(23*x[0]+29) % 671)).groupByKey().mapValues(list)
	h9 = h9.map(lambda x : (x[0], min(x[1])))

	h10 = data.map(lambda x : (x[1], 10*(17*x[0]+19) % 671)).groupByKey().mapValues(list)
	h10 = h10.map(lambda x : (x[0], min(x[1])))

	h11 = data.map(lambda x : (x[1], 11*(17*x[0]+19) % 671)).groupByKey().mapValues(list)
	h11 = h11.map(lambda x : (x[0], min(x[1])))

	h12 = data.map(lambda x : (x[1], 12*(13*x[0]+19) % 671)).groupByKey().mapValues(list)
	h12 = h12.map(lambda x : (x[0], min(x[1])))
	
	h13 = data.map(lambda x : (x[1], (3*x[0]+2) % 671)).groupByKey().mapValues(list)
	h13 = h13.map(lambda x : (x[0], min(x[1])))

	h14 = data.map(lambda x : (x[1], (5*x[0]+5) % 671)).groupByKey().mapValues(list)
	h14 = h14.map(lambda x : (x[0], min(x[1])))
	
	# Signature Matrix
	sm = h1.join(h2).mapValues(list)
	sm = sm.join(h3).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h4).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h5).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h6).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h7).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h8).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h9).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h10).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h11).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h12).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h13).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	sm = sm.join(h14).mapValues(list).map(lambda x : (x[0], append(x[1][0], x[1][1])))
	
	signatureMatrix = sm.sortByKey().collect()
	return signatureMatrix


start = time.time()
inputFile = sys.argv[1]

sc = SparkContext()
rdd = sc.textFile(inputFile, minPartitions=None, use_unicode=False)
rdd = rdd.mapPartitions(lambda x : csv.reader(x))
header = rdd.first()
rdd = rdd.filter(lambda x : x != header)

data = rdd.map(lambda x : (int(x[0])-1, int(x[1])))
signatureMatrix = getSignatureMatrix()

# LSH
N = 14
B = 7
R = 2

length = len(signatureMatrix)
candidatePairs = list()

for cur in range(length):
	curMovie = signatureMatrix[cur][0]
	curSignature = signatureMatrix[cur][1]
	for mov in range(cur+1, length):
		isCandidate = bandComparison(curSignature, signatureMatrix[mov][1], B, R)
		if isCandidate:
			pair = [curMovie, signatureMatrix[mov][0]]
			candidatePairs.append(pair)

# Jaccard similarity
dataDict = data.map(lambda x : (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()
pairsRdd = sc.parallelize(candidatePairs)
jaccardValues = pairsRdd.map(lambda x: (x, jaccardSimilarity(x[0], x[1])))
finalPairs = jaccardValues.filter(lambda x: x[1] >= 0.5)
finalList = finalPairs.collect()

# Writing to file
outputFile = "Nupur_Shukla_SimilarMovies_Jaccard.txt"
outFile = open(outputFile, "w")
finalList = sorted(finalList)
for i in range(1, len(finalList)):
	string = str(finalList[i][0][0]) + ", " + str(finalList[i][0][1]) + ", " + str(finalList[i][1]) + "\n"
	outFile.write(string)

'''
# Precision recall
pairsRdd = finalPairs.map(lambda x : (x[0][0], x[0][1]))
groundTruthFile = "/Users/nupur/Desktop/DataMining/Assignments/Assignment_03/data/SimilarMovies.GroundTruth.05.csv"
truthRdd = sc.textFile(groundTruthFile, minPartitions=None, use_unicode=False)
truthRdd = truthRdd.mapPartitions(lambda x : csv.reader(x))
truthRdd = truthRdd.map(lambda x : (int(x[0]), int(x[1])))

tp = float(pairsRdd.intersection(truthRdd).count())
fp = float(pairsRdd.subtract(truthRdd).count())
fn = float(truthRdd.subtract(pairsRdd).count())
precision = tp/(tp+fp)
recall = tp/(tp+fn)

print "Precision", precision
print "Recall", recall
'''
end = time.time()
print "Time: ", end - start, " seconds"
