# Movie-Recommendation-System

This work was done as part of INF-553 (Foundations and Applications of Data Mining) coursework at USC

<b>Environment requirements</b>- I have used Python 2.7 and Spark 2.2.1 to complete this assignment

Note: Paths can be relative to current directory or absolute.
Note: There should NOT be any spaces in the file path or file name <br/>

<b>Dataset</b>- Test data is present in data subfolder, and training data can be downloaded from https://grouplens.org/datasets/movielens/ (ml-latest-small.zip)

<b>Task 1</b>
>$SPARK_HOME/bin/spark-submit Nupur_Shukla_task1_Jaccard.py ml-latest-small/ratings.csv

I developed LSH (Local Sensitivity Hashing) technique to find similar movies according to the ratings of the users. I have focussed on 0-1 ratings rather than the actual ratings of the users. To be more specific, if a user has rated a movie, then his contribution to the characteristic matrix is 1 while if he hasn't rated a movie his contribution is 0. The goal is to identify similar movies whose Similarity is greater or equal to 0.5. <br/>
I have created 14 random hash functions to compute signature matrix. And I have used B=7 and R=2
Output file with name “Nupur_Shukla_SimilarMovies_Jaccard.txt” is created in the current directory. Following are the precision and recall values:<br/>
Precision 1.0
Recall 0.826665070413
Time:  99.0616660118  seconds

<b>Task 2: </b>
The file testing small.csv (20256 records) is from ratings.csv in ml-latest-small, correspondingly the file testing 20m.csv (4054451 records) is a subset of ratings.csv in ml-20m. The goal is to predict the ratings of every userId and movieId combination in the test files. I have implemented 3 techniques for the same: Model based CF, User based CF and Item based CF, and compared their performance in terms of time and accuracy.

<b>Model based CF</b>
>$SPARK_HOME/bin/spark-submit Nupur_Shukla_task2_ModelBasedCF.py /<Path of ratings.csv file/> /<Path of test.csv file/>

Example: $SPARK_HOME/bin/spark-submit Nupur_Shukla_task2_ModelBasedCF.py "ml-latest-small/ratings.csv" "data/testing_small.csv"

Parameters: rank = 10, iterations = 7, lambda_ = 0.1 

Output:<br/>
>=0 and <1:  13783
>=1 and <2:  4119
>=2 and <3:  717
>=3 and <4:  110
>=4:  4
RMSE:  0.952740143837
Time:  14.6660470963  sec

<b>User based CF</b>
>$SPARK_HOME/bin/spark-submit Nupur_Shukla_task2_UserBasedCF.py /<Path of ratings.csv file/> /<Path of test.csv file/>

Example: $SPARK_HOME/bin/spark-submit Nupur_Shukla_task2_UserBasedCF.py "ml-latest-small/ratings.csv" "data/testing_small.csv"

Used Pearson correlation similarity to compute weights and thus do the prediction using those.

Output:
>=0 and <1:  15122
>=1 and <2:  4194
>=2 and <3:  787
>=3 and <4:  142
>=4:  11
RMSE:  0.952949861361
Time:   190.558803082  sec

<b>Item based CF</b>
>$SPARK_HOME/bin/spark-submit Nupur_Shukla_task2_ItemBasedCF.py /<Path of ratings.csv file/> /<Path of test.csv file/>

Example: $SPARK_HOME/bin/spark-submit Nupur_Shukla_task2_ItemBasedCF.py "ml-latest-small/ratings.csv" "data/testing_small.csv"

Approach: I have implemented item-based CF with LSH and without LSH in the same python file. Results of item-based CF with LSH have been written to output file “Nupur_Shukla_task2_ItemBasedCF.txt”
Item based CF with LSH-
On running LSH first, we get pairs of most similar movie items. This is the pre-computation step. I have used the pairs I got from LSH, and computed Pearson Correlation weights only between those pairs to make the prediction for the test data.


Item based CF without LSH-
For this I have computed Pearson Correlation weights between all pairs of movie items, and used top 10 weights (i.e neighborhood value N = 10) and used these to make predictions for test data.

Results and Comparison:
Time taken to perform LSH = 100 sec

Itembased CF with LSH
>=0 and <1:  14723
>=1 and <2:  4483
>=2 and <3:  932
>=3 and <4:  117
>=4:  1
RMSE:  0.975430599377
Time to predict with LSH:  9.98718690872  sec

Itembased CF without LSH (N=10)
>=0 and <1:  13611
>=1 and <2:  5117
>=2 and <3:  1239
>=3 and <4:  249
>=4:  40
RMSE:  1.04775108034
Time to predict without LSH:  92.3959190845  sec

As it is visible from the results, item-based CF with LSH leads to much faster predictions than item-based CF without LSH. This is because with LSH we already get similar pairs, and thus we compute Pearson weights only between those pairs, as opposed to computing weights between all pairs (which is expensive because there are large number of items). This largely decreases the pair wise comparisons and leads to much faster predictions.

