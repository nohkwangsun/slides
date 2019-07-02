### Load Data
```scala
import org.apache.spark.ml.recommendation.ALS
val ratings = spark.read.textFile("/data/sample_movielens_ratings.txt")
  .selectExpr("split(value , '::') as col")
  .selectExpr(
    "cast(col[0] as int) as userId",
    "cast(col[1] as int) as movieId",
    "cast(col[2] as float) as rating",
    "cast(col[3] as long) as timestamp")
val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))
val als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
println(als.explainParams())
val alsModel = als.fit(training)
val predictions = alsModel.transform(test)
```
```console
alpha: alpha for implicit preference (default: 1.0)
checkpointInterval: set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext (default: 10)
coldStartStrategy: strategy for dealing with unknown or new users/items at prediction time. This may be useful in cross-validation or production scenarios, for handling user/item ids the model has not seen in the training data. Supported values: nan,drop. (default: nan)
finalStorageLevel: StorageLevel for ALS model factors. (default: MEMORY_AND_DISK)
implicitPrefs: whether to use implicit preference (default: false)
intermediateStorageLevel: StorageLevel for intermediate datasets. Cannot be 'NONE'. (default: MEMORY_AND_DISK)
itemCol: column name for item ids. Ids must be within the integer value range. (default: item, current: movieId)
maxIter: maximum number of iterations (>= 0) (default: 10, current: 5)
nonnegative: whether to use nonnegative constraint for least squares (default: false)
numItemBlocks: number of item blocks (default: 10)
numUserBlocks: number of user blocks (default: 10)
predictionCol: prediction column name (default: prediction)
rank: rank of the factorization (default: 10)
ratingCol: column name for ratings (default: rating, current: rating)
regParam: regularization parameter (>= 0) (default: 0.1, current: 0.01)
seed: random seed (default: 1994790107)
userCol: column name for user ids. Ids must be within the integer value range. (default: user, current: userId)
import org.apache.spark.ml.recommendation.ALS
ratings: org.apache.spark.sql.DataFrame = [userId: int, movieId: int ... 2 more fields]
training: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [userId: int, movieId: int ... 2 more fields]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [userId: int, movieId: int ... 2 more fields]
als: org.apache.spark.ml.recommendation.ALS = als_01f5bcdf6a33
alsModel: org.apache.spark.ml.recommendation.ALSModel = als_01f5bcdf6a33
predictions: org.apache.spark.sql.DataFrame = [userId: int, movieId: int ... 3 more fields]
```



### ALS
```scala
alsModel.recommendForAllUsers(10)
  .selectExpr("userId", "explode(recommendations)").show()
alsModel.recommendForAllItems(10)
  .selectExpr("movieId", "explode(recommendations)").show()
```
```console
+------+---------------+
|userId|            col|
+------+---------------+
|    28| [93, 6.575739]|
|    28| [81, 5.116676]|
|    28| [92, 4.990994]|
|    28|[10, 4.6011553]|
|    28| [40, 4.456021]|
|    28|[12, 4.4496927]|
|    28|  [9, 4.441847]|
|    28|[32, 4.4279385]|
|    28| [52, 4.419746]|
|    28|[74, 4.3055973]|
|    26|[94, 6.3221154]|
|    26| [24, 5.032882]|
|    26|[90, 4.9520164]|
|    26|  [7, 4.912031]|
|    26| [88, 4.796612]|
|    26| [68, 4.670635]|
|    26|[23, 4.5560145]|
|    26| [32, 4.422547]|
|    26| [17, 4.361108]|
|    26|[72, 4.2279005]|
+------+---------------+
only showing top 20 rows

+-------+---------------+
|movieId|            col|
+-------+---------------+
|     31|[12, 3.8703227]|
|     31|[16, 3.6222498]|
|     31|[22, 3.2817945]|
|     31|[14, 3.1975749]|
|     31|  [7, 2.954986]|
|     31|   [6, 2.90699]|
|     31|  [8, 2.818896]|
|     31| [15, 2.095858]|
|     31|[26, 1.9227128]|
|     31|[11, 1.5846266]|
|     85|  [16, 4.99182]|
|     85| [14, 4.814225]|
|     85|  [8, 4.656166]|
|     85|[22, 3.9892545]|
|     85|[18, 3.5882034]|
|     85|  [6, 3.269388]|
|     85|[24, 3.1881623]|
|     85|[26, 2.8329728]|
|     85| [3, 2.7982516]|
|     85| [21, 2.756485]|
+-------+---------------+
only showing top 20 rows
```

### RegressionEvaluator
```scala
import org.apache.spark.ml.evaluation.RegressionEvaluator
val evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")
```
```console
Root-mean-square error = 2.0005399337316847
import org.apache.spark.ml.evaluation.RegressionEvaluator
evaluator: org.apache.spark.ml.evaluation.RegressionEvaluator = regEval_637b88a2f9ea
rmse: Double = 2.0005399337316847
```

### RegressionMetrics
```scala
import org.apache.spark.mllib.evaluation.{
  RankingMetrics,
  RegressionMetrics}
val regComparison = predictions.select("rating", "prediction")
  .rdd.map(x => (x.getFloat(0).toDouble,x.getFloat(1).toDouble))
val metrics = new RegressionMetrics(regComparison)
```
```console
import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
regComparison: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[820] at map at <console>:41
metrics: org.apache.spark.mllib.evaluation.RegressionMetrics = org.apache.spark.mllib.evaluation.RegressionMetrics@55a94851
```

### predictions
```scala
import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.sql.functions.{col, expr}
val perUserActual = predictions
  .where("rating > 2.5")
  .groupBy("userId")
  .agg(expr("collect_set(movieId) as movies"))
```
```console
import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.sql.functions.{col, expr}
perUserActual: org.apache.spark.sql.DataFrame = [userId: int, movies: array<int>]
```

### predictions
```scala
val perUserPredictions = predictions
  .orderBy(col("userId"), col("prediction").desc)
  .groupBy("userId")
  .agg(expr("collect_list(movieId) as movies"))
```
```console
perUserPredictions: org.apache.spark.sql.DataFrame = [userId: int, movies: array<int>]
```

### RankingMetrics
```scala
val perUserActualvPred = perUserActual.join(perUserPredictions, Seq("userId"))
  .map(row => (
    row(1).asInstanceOf[Seq[Integer]].toArray,
    row(2).asInstanceOf[Seq[Integer]].toArray.take(15)
  ))
val ranks = new RankingMetrics(perUserActualvPred.rdd)
```
```console
perUserActualvPred: org.apache.spark.sql.Dataset[(Array[Integer], Array[Integer])] = [_1: array<int>, _2: array<int>]
ranks: org.apache.spark.mllib.evaluation.RankingMetrics[Integer] = org.apache.spark.mllib.evaluation.RankingMetrics@32c4c9b
```

### Result
```scala
ranks.meanAveragePrecision
ranks.precisionAt(5)
```
```console
res20: Double = 0.45384615384615384
```
