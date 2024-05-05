import mlflow
import json
import joblib
import click
from mlflow.tracking import MlflowClient
from mlflow.sklearn import load_model

def get_model(tracking_uri) :
     # Define your machine learning model class   
     mlflow.set_tracking_uri(tracking_uri)
     # tracking_uri = params['mlflow_config']['mlflow_tracking_uri']
     client = MlflowClient()
     # fetch model by model version
     model_details = client.get_model_version_by_alias(name = 'RFC Model', alias = 'production')
     # model = load_model(f'models:/{model_details.name}/{model_details.version}')
     # fetch model by model alias
     model = load_model(f"models:/{model_details.name}@{'production'}")
     return [model, model_details]

@click.command()
@click.argument('uri')
def save_model(uri) -> None:
     model, details = get_model(uri)
     joblib.dump(model, './model.joblib')
     
     with open('./model_details.json', 'w') as jsn :
          json.dump({'name': details.name,
                     'version': details.version,
                     'alias': details.aliases,
                     'runid': details.run_id
                    }, jsn)

if __name__ == '__main__' :
     save_model()
     