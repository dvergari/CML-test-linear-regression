import cml.data_v1 as cmldata
import pandas as pd
from pyspark import SparkContext


dataset = pd.read_excel('data.xlsx')
dataset.head()

# Needed because of https://github.com/YosefLab/Compass/issues/92
pd.DataFrame.iteritems = pd.DataFrame.items

CONNECTION_NAME = "ps-aw-dl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

sparkdf = spark.createDataFrame(dataset)

sparkdf.count()

sparkdf.createOrReplaceTempView("pp")
spark.sql("drop table if exists default.power_plant")
spark.sql("create table power_plant using iceberg as select * from pp")