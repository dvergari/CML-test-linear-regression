import cml.data_v1 as cmldata
from pyspark import SparkContext

import mlflow

from pyspark.ml.feature import VectorAssembler

CONNECTION_NAME = "ps-aw-dl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

power_plant_df = spark.read.table("default.power_plant")

vectorAssembler = VectorAssembler(inputCols = 
                                  ['AT', 'V', 'AP', 'RH'], 
                                  outputCol = 'features')

vpower_plant = vectorAssembler.transform(power_plant_df)

vpower_plant = vpower_plant.select(['features', 'PE'])

splits = vpower_plant.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'features', 
                      labelCol='PE', maxIter=10, regParam=0.1, 
                      elasticNetParam=0.99)

mlflow.set_experiment("Linear regression - spark")

with mlflow.start_run():

  mlflow.log_param('maxIter',10)
  mlflow.log_param('regParam',0.1)
  mlflow.log_param('elasticNetParam',0.99)


  lr_model = lr.fit(train_df)
  print("Coefficients: " + str(lr_model.coefficients))
  print("Intercept: " + str(lr_model.intercept))

  trainingSummary = lr_model.summary
  print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
  print("r2: %f" % trainingSummary.r2)

  test_result = lr_model.evaluate(test_df)
  print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)

  print("numIterations: %d" % trainingSummary.totalIterations)
  print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
  trainingSummary.residuals.show()

  predictions = lr_model.transform(test_df)
  predictions.select("prediction","PE","features").show()
  mlflow.log_metric("RMSE", test_result.rootMeanSquaredError)
  mlflow.log_metric("r2", test_result.r2)

  # Use Spark model.save() 
  lr_model.write().overwrite().save("s3a://ps-uat2/data/mymodel")

  
# Test inference
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler

loaded_model = LinearRegressionModel.load("s3a://ps-uat2/data/mymodel")

df = spark.read.json(spark.sparkContext.parallelize(['{"AP":2000,"AT":15,"RH":75,"V":40}']))

vectorAssembler = VectorAssembler(inputCols = 
                                    ['AT', 'V', 'AP', 'RH'], 
                                    outputCol = 'features')


df1 = vectorAssembler.transform(df)
df1.toJSON().first()