import cml.data_v1 as cmldata
from pyspark import SparkContext

import mlflow

from pyspark.ml.feature import VectorAssembler

CONNECTION_NAME = "ps-aw-dl"
MODEL_NAME = "ElasticNetPowerPlant"

MAX_ITER=100
REG_PARAM=0.3
ELASTIC_NET_PARAM=0.8

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
                      labelCol='PE', maxIter=MAX_ITER, regParam=REG_PARAM, 
                      elasticNetParam=ELASTIC_NET_PARAM)

mlflow.set_experiment("Linear regression")

with mlflow.start_run():

  mlflow.log_param('maxIter',MAX_ITER)
  mlflow.log_param('regParam',REG_PARAM)
  mlflow.log_param('elasticNetParam',ELASTIC_NET_PARAM)


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
  mlflow.spark.log_model(lr_model, "model", registered_model_name=MODEL_NAME)

  