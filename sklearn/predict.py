import mlflow
import json
import pandas as pd

logged_model = '/home/cdsw/.experiments/fdak-cd5w-c51v-24kq/6nxg-681h-b5pe-cyqy/artifacts/model'


def predict(args):
  loaded_model = mlflow.pyfunc.load_model(logged_model)
  #df = pd.json_normalize(data['inputs'])
  return loaded_model.predict(pd.DataFrame(args))

  