### Load Data
```scala
import org.apache.spark.ml.feature.VectorAssembler

val va = new VectorAssembler()
  .setInputCols(Array("Quantity", "UnitPrice"))
  .setOutputCol("features")

val sales = va.transform(spark.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/data/retail-data/by-day/*.csv")
  .limit(50)
  .coalesce(1)
  .where("Description IS NOT NULL"))

sales.cache()
```
```console
import org.apache.spark.ml.feature.VectorAssembler
va: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_34bf9e20439c
sales: org.apache.spark.sql.DataFrame = [InvoiceNo: string, StockCode: string ... 7 more fields]
res2: sales.type = [InvoiceNo: string, StockCode: string ... 7 more fields]
```

### KMeans
```scala
import org.apache.spark.ml.clustering.KMeans
val km = new KMeans().setK(5)
println(km.explainParams())
val kmModel = km.fit(sales)
```
```console
distanceMeasure: The distance measure. Supported options: 'euclidean' and 'cosine' (default: euclidean)
featuresCol: features column name (default: features)
initMode: The initialization algorithm. Supported options: 'random' and 'k-means||'. (default: k-means||)
initSteps: The number of steps for k-means|| initialization mode. Must be > 0. (default: 2)
k: The number of clusters to create. Must be > 1. (default: 2, current: 5)
maxIter: maximum number of iterations (>= 0) (default: 20)
predictionCol: prediction column name (default: prediction)
seed: random seed (default: -1689246527)
tol: the convergence tolerance for iterative algorithms (>= 0) (default: 1.0E-4)
import org.apache.spark.ml.clustering.KMeans
km: org.apache.spark.ml.clustering.KMeans = kmeans_b640586f900a
kmModel: org.apache.spark.ml.clustering.KMeansModel = kmeans_b640586f900a
```

### Summary of KMeans
```scala
val summary = kmModel.summary
summary.clusterSizes.foreach(println) // number of points
kmModel.computeCost(sales)
println("Cluster Centers: ")
kmModel.clusterCenters.foreach(println)
```
```console
warning: there was one deprecation warning; re-run with -deprecation for details
10
20
12
3
5
Cluster Centers: 
[23.200000000000003,0.9560000000000001]
[4.55,4.5965]
[11.333333333333332,1.0999999999999996]
[44.0,1.1633333333333333]
[2.4000000000000004,13.040000000000001]
summary: org.apache.spark.ml.clustering.KMeansSummary = org.apache.spark.ml.clustering.KMeansSummary@21478bc4
```

### BiSectingKmeans
```scala
import org.apache.spark.ml.clustering.BisectingKMeans
val bkm = new BisectingKMeans().setK(5).setMaxIter(5)
println(bkm.explainParams())
val bkmModel = bkm.fit(sales)
```
```console
distanceMeasure: The distance measure. Supported options: 'euclidean' and 'cosine' (default: euclidean)
featuresCol: features column name (default: features)
k: The desired number of leaf clusters. Must be > 1. (default: 4, current: 5)
maxIter: maximum number of iterations (>= 0) (default: 20, current: 5)
minDivisibleClusterSize: The minimum number of points (if >= 1.0) or the minimum proportion of points (if < 1.0) of a divisible cluster. (default: 1.0)
predictionCol: prediction column name (default: prediction)
seed: random seed (default: 566573821)
import org.apache.spark.ml.clustering.BisectingKMeans
bkm: org.apache.spark.ml.clustering.BisectingKMeans = bisecting-kmeans_f34c812e88b1
bkmModel: org.apache.spark.ml.clustering.BisectingKMeansModel = bisecting-kmeans_f34c812e88b1
```

## Summary of BiSectingKmeans
```scala
val summary = bkmModel.summary
summary.clusterSizes.foreach(println) // number of points
kmModel.computeCost(sales)
println("Cluster Centers: ")
kmModel.clusterCenters.foreach(println)
```
```console
warning: there was one deprecation warning; re-run with -deprecation for details
16
8
13
10
3
Cluster Centers: 
[23.200000000000003,0.9560000000000001]
[4.55,4.5965]
[11.333333333333332,1.0999999999999996]
[44.0,1.1633333333333333]
[2.4000000000000004,13.040000000000001]
summary: org.apache.spark.ml.clustering.BisectingKMeansSummary = org.apache.spark.ml.clustering.BisectingKMeansSummary@7d18bdd8
```

### GMM
```scala
import org.apache.spark.ml.clustering.GaussianMixture
val gmm = new GaussianMixture().setK(5)
println(gmm.explainParams())
val model = gmm.fit(sales)
```
```console
featuresCol: features column name (default: features)
k: Number of independent Gaussians in the mixture model. Must be > 1. (default: 2, current: 5)
maxIter: maximum number of iterations (>= 0) (default: 100)
predictionCol: prediction column name (default: prediction)
probabilityCol: Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities (default: probability)
seed: random seed (default: 538009335)
tol: the convergence tolerance for iterative algorithms (>= 0) (default: 0.01)
import org.apache.spark.ml.clustering.GaussianMixture
gmm: org.apache.spark.ml.clustering.GaussianMixture = GaussianMixture_448cbe438110
model: org.apache.spark.ml.clustering.GaussianMixtureModel = GaussianMixture_448cbe438110
```

### Summary of GMM
```scala
val summary = model.summary
model.weights.foreach(println)
model.gaussiansDF.show()
summary.cluster.show()
summary.clusterSizes
summary.probability.show()
```
```console
0.2037672399608971
0.13999983831435028
0.1633437562383104
0.1755724681410311
0.3173166973454111
+--------------------+--------------------+
|                mean|                 cov|
+--------------------+--------------------+
|[26.8969402794066...|155.4804310513139...|
|[10.8571461378584...|3.265300498555194...|
|[2.52584147369065...|0.770164672724497...|
|[17.7437989010750...|39.65470455661214...|
|[4.82603980059328...|1.655630468857532...|
+--------------------+--------------------+

+----------+
|prediction|
+----------+
|         0|
|         0|
|         0|
|         0|
|         4|
|         0|
|         4|
|         0|
|         4|
|         4|
|         4|
|         4|
|         1|
|         1|
|         3|
|         3|
|         0|
|         0|
|         4|
|         4|
+----------+
only showing top 20 rows

+--------------------+
|         probability|
+--------------------+
|[0.99999999999717...|
|[0.99997354327726...|
|[0.99999999999985...|
|[0.99999999999963...|
|[0.00553284170817...|
|[0.99999999999929...|
|[8.19503968942260...|
|[0.99999999999983...|
|[5.77416968211996...|
|[5.77416968211996...|
|[7.83213162884820...|
|[1.40641820378790...|
|[2.59516003767789...|
|[2.59516003767789...|
|[0.08975856024991...|
|[0.21615743245013...|
|[0.99999999999966...|
|[0.99999999999963...|
|[5.77416968211996...|
|[5.77416968211996...|
+--------------------+
only showing top 20 rows

summary: org.apache.spark.ml.clustering.GaussianMixtureSummary = org.apache.spark.ml.clustering.GaussianMixtureSummary@4ebf3887
```

### CountVectorize for LDA
```scala
import org.apache.spark.ml.feature.{Tokenizer, CountVectorizer}
val tkn = new Tokenizer().setInputCol("Description").setOutputCol("DescOut")
val tokenized = tkn.transform(sales.drop("features"))
val cv = new CountVectorizer()
  .setInputCol("DescOut")
  .setOutputCol("features")
  .setVocabSize(500)
  .setMinTF(0)
  .setMinDF(0)
  .setBinary(true)
val cvFitted = cv.fit(tokenized)
val prepped = cvFitted.transform(tokenized)
```
```console
import org.apache.spark.ml.feature.{Tokenizer, CountVectorizer}
tkn: org.apache.spark.ml.feature.Tokenizer = tok_e163bc71ac63
tokenized: org.apache.spark.sql.DataFrame = [InvoiceNo: string, StockCode: string ... 7 more fields]
cv: org.apache.spark.ml.feature.CountVectorizer = cntVec_ea984e77adb0
cvFitted: org.apache.spark.ml.feature.CountVectorizerModel = cntVec_ea984e77adb0
prepped: org.apache.spark.sql.DataFrame = [InvoiceNo: string, StockCode: string ... 8 more fields]
```

### LDA
```scala
import org.apache.spark.ml.clustering.LDA
val lda = new LDA().setK(10).setMaxIter(5)
println(lda.explainParams())
val model = lda.fit(prepped)
```
```console
checkpointInterval: set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext (default: 10)
docConcentration: Concentration parameter (commonly named "alpha") for the prior placed on documents' distributions over topics ("theta"). (undefined)
featuresCol: features column name (default: features)
k: The number of topics (clusters) to infer. Must be > 1. (default: 10, current: 10)
keepLastCheckpoint: (For EM optimizer) If using checkpointing, this indicates whether to keep the last checkpoint. If false, then the checkpoint will be deleted. Deleting the checkpoint can cause failures if a data partition is lost, so set this bit with care. (default: true)
learningDecay: (For online optimizer) Learning rate, set as an exponential decay rate. This should be between (0.5, 1.0] to guarantee asymptotic convergence. (default: 0.51)
learningOffset: (For online optimizer) A (positive) learning parameter that downweights early iterations. Larger values make early iterations count less. (default: 1024.0)
maxIter: maximum number of iterations (>= 0) (default: 20, current: 5)
optimizeDocConcentration: (For online optimizer only, currently) Indicates whether the docConcentration (Dirichlet parameter for document-topic distribution) will be optimized during training. (default: true)
optimizer: Optimizer or inference algorithm used to estimate the LDA model. Supported: online, em (default: online)
seed: random seed (default: 1435876747)
subsamplingRate: (For online optimizer) Fraction of the corpus to be sampled and used in each iteration of mini-batch gradient descent, in range (0, 1]. (default: 0.05)
topicConcentration: Concentration parameter (commonly named "beta" or "eta") for the prior placed on topic' distributions over terms. (undefined)
topicDistributionCol: Output column with estimates of the topic mixture distribution for each document (often called "theta" in the literature).  Returns a vector of zeros for an empty document. (default: topicDistribution)
import org.apache.spark.ml.clustering.LDA
lda: org.apache.spark.ml.clustering.LDA = lda_1d8684c4a71a
model: org.apache.spark.ml.clustering.LDAModel = lda_1d8684c4a71a
```

### describeTopics of LDA
```scala
model.describeTopics(3).show()
cvFitted.vocabulary
```
```console
+-----+---------------+--------------------+
|topic|    termIndices|         termWeights|
+-----+---------------+--------------------+
|    0|   [69, 88, 81]|[0.00908107405013...|
|    1|[107, 127, 100]|[0.00888050121255...|
|    2|  [15, 77, 120]|[0.01734248131232...|
|    3|    [36, 72, 9]|[0.01159491697518...|
|    4|  [137, 34, 22]|[0.00905035557253...|
|    5|  [34, 124, 95]|[0.00911696062041...|
|    6|     [2, 7, 18]|[0.01756473154171...|
|    7|   [87, 58, 73]|[0.00867808805804...|
|    8|      [1, 3, 0]|[0.01801302800320...|
|    9|  [29, 132, 34]|[0.00954482252355...|
+-----+---------------+--------------------+

res6: Array[String] = Array(water, hot, vintage, bottle, paperweight, 6, home, doormat, landmark, bicycle, frame, ribbons, "", classic, rose, kit, leaf, sweet, bag, airline, doorstop, light, in, christmas, heart, calm, set, keep, balloons, night, lights, 12, tin, english, caravan, stuff, tidy, oxford, full, cottage, notting, drawer, mushrooms, chrome, champion, amelie, mini, the, giant, design, elegant, tins, jet, fairy, 50's, holder, message, blue, storage, tier, covent, world, skulls, font, hearts, skull, clips, bell, red, party, chalkboard, save, 4, coloured, poppies, garden, nine, girl, shimmering, doughnut, dog, 3, tattoos, chilli, coat, torch, sunflower, tale, cards, puncture, woodland, bomb, knack, lip, collage, rabbit, sex, of, rack, wall, cracker, scottie, hill, led, black, art...
```

