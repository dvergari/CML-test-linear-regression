# Linear Regression
import mlflow
import pandas as pd
dataset = pd.read_excel('data.xlsx')
dataset.head()

REG_MODEL_NAME = "ElasticNetPowerPlantSklearn"

# Getting the inputs and output
X = dataset.iloc[:, :-1].values
X
y = dataset.iloc[:, -1].values
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train
X_test
y_train
y_test

# Part 2 - Building and training the model

# Building the model
from sklearn.linear_model import LinearRegression

mlflow.set_experiment("Linear Regression Sklearn")
mlflow.autolog()
model = LinearRegression()

# Training the model
with mlflow.start_run():

  model.fit(X_train, y_train)

  # Inference

  # Making the predictions of the data points in the test set
  y_pred = model.predict(X_test)
  y_pred

  # Making the prediction of a single data point with AT = 15, V = 40, AP = 1000, RH = 75
  model.predict([[15,40,1000,75]])

  # Part 3: Evaluating the model

  # R-Squared
  from sklearn.metrics import r2_score
  r2 = r2_score(y_test, y_pred)
  r2

  # Adjusted R-Squared
  k = X_test.shape[1]
  n = X_test.shape[0]
  adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)
  adj_r2
  mlflow.sklearn.log_model(model, "model", registered_model_name=REG_MODEL_NAME)
