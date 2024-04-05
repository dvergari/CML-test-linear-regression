import cmlapi
import time
import os
import json


api_client = cmlapi.default_client()

REG_MODEL_NAME = "ElasticNetPowerPlant"
MODEL_NAME = "linear-regression"

def get_current_project_id():
  return os.environ['CDSW_PROJECT_ID']

def get_registered_model_id(model_name):
  reg_models = api_client.list_registered_models(
    search_filter= json.dumps({"model_name": REG_MODEL_NAME}), page_size=100 )
  return reg_models.models[0].model_id

def get_latest_version_id(model_id):
  model_versions = api_client.get_registered_model(reg_model_id, sort="-version_number")
  return model_versions.model_versions[0].model_version_id

project_id = get_current_project_id()

reg_model_id = get_registered_model_id(REG_MODEL_NAME)


model_body = cmlapi.CreateModelRequest(
  project_id=project_id,
  name=MODEL_NAME,
  description="Linear Regression Test", 
  disable_authentication=True,
  registered_model_id=reg_model_id
)

model = ""
try:
  model = api_client.list_models(project_id, 
                                 search_filter=json.dumps({"name":MODEL_NAME})).models[0]
except IndexError:
  model = api_client.create_model(model_body, project_id)

model_build_body = cmlapi.CreateModelBuildRequest(
  project_id=project_id,
  model_id=model.id,
  kernel="python3",
  runtime_identifier='docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-workbench-python3.10-standard:2024.02.1-b4', 
  registered_model_version_id=get_latest_version_id(reg_model_id)
)

model_build = api_client.create_model_build(model_build_body, project_id, model.id)


while model_build.status not in ["built", "build failed"]:
  print("waiting for model to build...")
  time.sleep(10)
  model_build = api_client.get_model_build(project_id, model.id, model_build.id)

if model_build.status == "build failed":
  print("model build failed, see UI for more information")
  sys.exit(1)

print("model built successfully!")

model_replicas=1

model_deployment_body = cmlapi.CreateModelDeploymentRequest(
  project_id=project_id, 
  model_id=model.id, 
  build_id=model_build.id, 
  replicas = model_replicas)

model_deployment = api_client.create_model_deployment(model_deployment_body,
                                   project_id,
                                   model.id,
                                   model_build.id)

while model_deployment.status not in ["stopped", "failed", "deployed"]:
  print("waiting for model to deploy...")
  time.sleep(10)
  model_deployment = api_client.get_model_deployment(project_id, model.id, model_build.id, model_deployment.id)

if model_deployment.status != "deployed":
  print("model deployment failed, see UI for more information")
  sys.exit(1)

print("model deployed successfully!")