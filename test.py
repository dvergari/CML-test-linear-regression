import cml.data_v1 as cmldata

CONNECTION_NAME = "ps-aw-dl"
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()