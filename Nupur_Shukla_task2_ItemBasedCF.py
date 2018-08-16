from pyspark import SparkContext
import csv
import sys
import time
import math

def categoriesForPrinting(categories):
	cat0 = 0
	cat1 = 0
	cat2 = 0
	cat3 = 0
	cat4 = 0
	for category in categories:
		cat = category[0]
		if cat == 0:
			cat0 = category[1]
		elif cat == 1:
			cat1 = category[1]
		elif cat == 2:
			cat2 = category[1]
		elif cat == 3:
			cat3 = category[1]
		elif cat == 4:
			cat4 = category[1]
	return cat0, cat1, cat2, cat3, cat4

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

def predict(activeUser, activeMovie, topSimilarMovies):

	# average of active user
	activeUserData = usersRdd[activeUser]
	a_temp = [x[1] for x in activeUserData]
	a_mean = sum(a_temp)/len(a_temp)

	topSimilarMovies = list(set(topSimilarMovies))

	if len(topSimilarMovies) == 0:
		return a_mean

	if(len(topSimilarMovies) == 1) and (topSimilarMovies[0][1] == activeMovie):
		return a_mean

	else:
		num = 0
		den = 0
		for topSimilarMovie in topSimilarMovies:
			otherMovie = topSimilarMovie[1]

			key = (activeUser, otherMovie)
			if key in userMovieRdd:
				# check condition
				o_rating = userMovieRdd[(activeUser, otherMovie)]
				den = den + abs(topSimilarMovie[0])
				num = num + topSimilarMovie[0]*o_rating

		if den == 0 or num == 0:
			return a_mean
		else:
			pred = num/den
			if pred < 0:
				pred = -1 * pred
			return pred

def pearsonCorrelation(activeMovieData, otherMovieData):
	corrated = list()
	i = 0
	j = 0
	activeMovieData.sort()
	otherMovieData.sort()
	while (i<len(activeMovieData) and j< len(otherMovieData)):
		if activeMovieData[i][0] == otherMovieData[j][0]:
			corrated.append((activeMovieData[i][0], (activeMovieData[i][1], otherMovieData[j][1])))
			i = i+1
			j = j+1

		elif activeMovieData[i][0] < otherMovieData[j][0]:
			i = i+1

		else:
			j = j+1

	if len(corrated) == 0 or len(corrated) == 1:
		# no corrated items or only 1 corrated items
		return -2.0

	active = [x[1][0] for x in corrated]
	a_mean = sum(active)/len(active)

	other = [x[1][1] for x in corrated]
	o_mean = sum(active)/len(active)

	a_list = active
	o_list = other

	num = 0.0
	d1 = 0.0
	d2 = 0.0
	for i in range(len(a_list)):
		a = a_list[i] - a_mean
		o = o_list[i] - o_mean
		num = num + a*o
		d1 = d1 + (a*a)
		d2 = d2 + (o*o)

	den = math.sqrt(d1) * math.sqrt(d2)
	if den == 0 or num == 0:
		return -2.0
	else:
		return num/den

def getSimilarMoviesWithoutLSH(activeUser, activeMovie):
	topSimilarMovies = list()
	if activeMovie not in moviesRdd:
		# item never rated by any user (item cold start)
		topSimilarMovies.append((1, activeMovie))
		return topSimilarMovies

	activeMovieData = moviesRdd[activeMovie]

	activeUserData = usersRdd[activeUser]
	otherMovies = [x[0] for x in activeUserData]
	for otherMovie in otherMovies:
		if activeMovie != otherMovie:
			otherMovieData = moviesRdd[otherMovie]
			similarity = pearsonCorrelation(activeMovieData, otherMovieData)
			if similarity != -2.0:
				topSimilarMovies.append((similarity, otherMovie))
			else:
				topSimilarMovies.append((1, activeMovie))

	similarMovies = sorted(topSimilarMovies, reverse=True)
	return similarMovies[:10] #Neighbourhood value 10

def getSimilarMoviesWithLSH(activeUser, activeMovie):
	topSimilarMovies = list()
	if (activeMovie not in similarPairs1) and (activeMovie not in similarPairs2):
		# no movie similar to active movie
		# assign average rating of user
		topSimilarMovies.append((1, activeMovie))
		return topSimilarMovies

	if activeMovie not in moviesRdd:
		# item never rated by any user (item cold start)
		topSimilarMovies.append((1, activeMovie))
		return topSimilarMovies

	activeMovieData = moviesRdd[activeMovie]
	otherMovies = list()
	if activeMovie in similarPairs1:
		otherMovies = otherMovies + similarPairs1[activeMovie]
	if activeMovie in similarPairs2:
		otherMovies = otherMovies + similarPairs2[activeMovie]

	otherMovies = list(set(otherMovies))

	for otherMovie in otherMovies:
		if activeMovie != otherMovie:
			otherMovieData = moviesRdd[otherMovie]
			similarity = pearsonCorrelation(activeMovieData, otherMovieData)
			if similarity != -2.0:
				topSimilarMovies.append((similarity, otherMovie))
			else:
				topSimilarMovies.append((1, activeMovie))

	similarMovies = sorted(topSimilarMovies, reverse=True)
	return similarMovies

def jaccardSimilarity(dataDict, movie1, movie2):
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

def performLSH():
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
	jaccardValues = pairsRdd.map(lambda x: (x, jaccardSimilarity(dataDict, x[0], x[1])))
	finalPairsAndSimilarity = jaccardValues.filter(lambda x: x[1] >= 0.5)
	return finalPairsAndSimilarity

start = time.time()
ratingsFile = sys.argv[1]
testFile = sys.argv[2]

sc = SparkContext()
ratingsRdd = sc.textFile(ratingsFile, minPartitions=None, use_unicode=False).mapPartitions(lambda x : csv.reader(x))
header1 = ratingsRdd.first()
ratingsRdd = ratingsRdd.filter(lambda x : x != header1)
ratingsRdd = ratingsRdd.map(lambda x : ((int(x[0]), int(x[1])), float(x[2])))

testRdd = sc.textFile(testFile, minPartitions=None, use_unicode=False).mapPartitions(lambda x : csv.reader(x))
header2 = testRdd.first()
testRdd = testRdd.filter(lambda x : x != header2)
testRdd = testRdd.map(lambda x : ((int(x[0]), int(x[1])), 1))

trainingRdd = ratingsRdd.subtractByKey(testRdd)
trainingRdd = trainingRdd.map(lambda x : (x[0][0], (x[0][1], x[1])))
testRdd = testRdd.map(lambda x : (x[0][0], x[0][1])).sortByKey()

usersRdd = trainingRdd.groupByKey().sortByKey().mapValues(list).collectAsMap()
userMovieRdd = trainingRdd.map(lambda x : ((x[0], x[1][0]), x[1][1])).sortByKey().collectAsMap()
moviesRdd = trainingRdd.map(lambda x : (x[1][0], (x[0], x[1][1]))).groupByKey().sortByKey().mapValues(list).collectAsMap()

# Perform LSH
data = trainingRdd.map(lambda x : (x[0]-1, x[1][0]))
finalPairs = performLSH().collect()

# Prediction with LSH
t1_start = time.time()
similarPairs1 = sc.parallelize(finalPairs).map(lambda x : (x[0][0], x[0][1])).groupByKey().sortByKey().mapValues(list).collectAsMap()
similarPairs2 = sc.parallelize(finalPairs).map(lambda x : (x[0][1], x[0][0])).groupByKey().sortByKey().mapValues(list).collectAsMap()
topSimilarMovies = testRdd.map(lambda x : (x[0], x[1], getSimilarMoviesWithLSH(x[0], x[1])))
predictions = topSimilarMovies.map(lambda x: (x[0], x[1], predict(x[0], x[1], x[2])))

# Write with-LSH results to file
predictionsList = predictions.collect()
outFile = open("Nupur_Shukla_task2_ItemBasedCF.txt", "w")
for p in predictionsList:
	outFile.write(str(p[0]) + ", " + str(p[1]) + ", " + str(p[2]) + "\n")
outFile.close()

results = predictions.map(lambda x : ((x[0], x[1]), x[2])).join(ratingsRdd)
differences = results.map(lambda x: abs(x[1][0]-x[1][1]))
categoriesWithLSH = differences.map(lambda x : (category(x), 1)).reduceByKey(lambda x, y : x+y).sortByKey().collect()
rmseWithLSH = math.sqrt(differences.map(lambda x: x**2).mean())
t1_end = time.time()

# Prediction without LSH
t2_start = time.time()
topSimilarMovies1 = testRdd.map(lambda x : (x[0], x[1], getSimilarMoviesWithoutLSH(x[0], x[1])))
predictionsWithoutLSH = topSimilarMovies1.map(lambda x: (x[0], x[1], predict(x[0], x[1], x[2])))
results = predictionsWithoutLSH.map(lambda x : ((x[0], x[1]), x[2])).join(ratingsRdd)
differences = results.map(lambda x: abs(x[1][0]-x[1][1]))
categoriesWithoutLSH = differences.map(lambda x : (category(x), 1)).reduceByKey(lambda x, y : x+y).sortByKey().collect()
rmseWithoutLSH = math.sqrt(differences.map(lambda x: x**2).mean())
t2_end = time.time()

# Printing all results
cat0, cat1, cat2, cat3, cat4 = categoriesForPrinting(categoriesWithLSH)
print "Itembased CF with LSH"
print ">=0 and <1: ", cat0
print ">=1 and <2: ", cat1
print ">=2 and <3: ", cat2
print ">=3 and <4: ", cat3
print ">=4: ", cat4
print "RMSE: ", str(rmseWithLSH)
print "Time to predict with LSH: ", t1_end - t1_start, " sec"

cat0, cat1, cat2, cat3, cat4 = categoriesForPrinting(categoriesWithoutLSH)
print "\nItembased CF without LSH"
print ">=0 and <1: ", cat0
print ">=1 and <2: ", cat1
print ">=2 and <3: ", cat2
print ">=3 and <4: ", cat3
print ">=4: ", cat4
print "RMSE: ", str(rmseWithoutLSH)
print "Time to predict without LSH: ", t2_end - t2_start, " sec"

end = time.time()