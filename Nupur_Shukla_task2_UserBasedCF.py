from pyspark import SparkContext
import csv
import sys
import time
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

def predict(activeUser, activeMovie, topSimilarUsers):
	activeUserData = usersRdd[activeUser]
	a_temp = [x[1] for x in activeUserData]
	a_mean = sum(a_temp)/len(a_temp)

	if len(topSimilarUsers) == 0:
		return a_mean

	else:
		num = 0
		den = 0
		for topSimilarUser in topSimilarUsers:
			otherUser = topSimilarUser[1]

			key = (otherUser, activeMovie)
			if key in userMovieRdd:
				# check condition
				o_rating = userMovieRdd[(otherUser, activeMovie)]

				otherData = usersRdd[otherUser]
				o_temp = [x[1] for x in otherData if otherData[0] != activeMovie]

				o_mean = sum(o_temp)/len(o_temp)
				den = den + abs(topSimilarUser[0])
				num = num + topSimilarUser[0]*(o_rating - o_mean)

		if den == 0:
			return a_mean
		else:
			return (a_mean + num/den)


def pearsonCorrelation(activeUserData, otherUserData):
	corrated = list()
	i = 0
	j = 0
	activeUserData.sort()
	otherUserData.sort()
	while (i<len(activeUserData) and j< len(otherUserData)):
		if activeUserData[i][0] == otherUserData[j][0]:
			corrated.append((activeUserData[i][0], (activeUserData[i][1], otherUserData[j][1])))
			i = i+1
			j = j+1

		elif activeUserData[i][0] < otherUserData[j][0]:
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


def getSimilarUsers(activeUser, activeMovie):
	similarUsers = list()

	if activeMovie not in moviesRdd:
		# item cold start
		similarUsers.append((0, activeUser))
		return similarUsers

	activeUserData = usersRdd[activeUser]
	otherUsers = moviesRdd[activeMovie]

	for otherUser in otherUsers:
		if activeUser != otherUser:
			otherUserData = usersRdd[otherUser]
			similarity = pearsonCorrelation(activeUserData, otherUserData)
			if similarity != -2.0:
				similarUsers.append((similarity, otherUser))

	similarUsers = sorted(similarUsers, reverse=True)
	return similarUsers


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

userMovieRdd = trainingRdd.map(lambda x : ((x[0], x[1][0]), x[1][1])).sortByKey().collectAsMap()
usersRdd = trainingRdd.groupByKey().sortByKey().mapValues(list).collectAsMap()
moviesRdd = trainingRdd.map(lambda x : (x[1][0], x[0])).groupByKey().sortByKey().mapValues(list).collectAsMap()

topSimilarUsers = testRdd.map(lambda x : (x[0], x[1], getSimilarUsers(x[0], x[1])))
predictions = topSimilarUsers.map(lambda x: (x[0], x[1], predict(x[0], x[1], x[2])))

predictionsList = sorted(predictions.collect())
outFile = open("Nupur_Shukla_task2_UserBasedCF.txt", "w")
for p in predictionsList:
	outFile.write(str(p[0]) + ", " + str(p[1]) + ", " + str(p[2]) + "\n")
outFile.close()

results = predictions.map(lambda x : ((x[0], x[1]), x[2])).join(ratingsRdd)

differences = results.map(lambda x: abs(x[1][0]-x[1][1]))
categories = differences.map(lambda x : (category(x), 1)).reduceByKey(lambda x, y : x+y).sortByKey().collect()
rmse = math.sqrt(differences.map(lambda x: x**2).mean())

print ">=0 and <1: ",categories[0][1]
print ">=1 and <2: ",categories[1][1]
print ">=2 and <3: ",categories[2][1]
print ">=3 and <4: ",categories[3][1]
print ">=4: ",categories[4][1]
print "RMSE: ", str(rmse)

end = time.time()
print "Time: ", end - start, " sec"