import mlflow
from pyspark.ml.feature import VectorAssembler


logged_model = '/home/cdsw/.experiments/q77y-jgmv-r63o-cp71/1eu0-xmw6-he28-o2iz/artifacts/model'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

df = spark.read.json(spark.sparkContext.parallelize(['{"AP":2000,"AT":15,"RH":75,"V":40}']))

vectorAssembler = VectorAssembler(inputCols = 
                                    ['AT', 'V', 'AP', 'RH'], 
                                    outputCol = 'features')


df1 = vectorAssembler.transform(df)

# Predict on a Spark DataFrame.
df1.withColumn('predictions', loaded_model("AP","AT","RH","V")).collect()