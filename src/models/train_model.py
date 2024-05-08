import pathlib
import yaml
import joblib
import mlflow
import typing
import numpy as np
import pandas as pd
from mlflow.sklearn import log_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.base import BaseEstimator
from src.logger import infologger

infologger.info('*** Executing: train_model.py ***')
# writing import after infologger to log the info precisely 
from src.visualization import visualize
from src.data.make_dataset import load_data

def train_model(training_feat: np.ndarray, y_true: pd.Series, n_estimators: int, criterion: str, max_depth: int, min_samples_split: int, min_samples_leaf: int, random_state: int, yaml_file_obj: typing.IO) -> dict :
     try : 
          model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth, min_samples_split = min_samples_split,
                                         min_samples_leaf = min_samples_leaf, random_state = random_state)
          model.fit(training_feat, y_true)
     except Exception as e :
          infologger.info(f'unable to fit RFC model [check train_model()]. exc: {e}')
     else :
          infologger.info(f'trained {type(model).__name__} model')
          y_pred = model.predict(training_feat)
          y_pred_prob = model.predict_proba(training_feat)
          accuracy = round(metrics.balanced_accuracy_score(y_true, y_pred), 5)
          precision = round(metrics.precision_score(y_true, y_pred, zero_division = 1, average = 'macro'), 5)
          recall = round(metrics.recall_score(y_true, y_pred, average = 'macro'), 5)
          roc_score = round(metrics.roc_auc_score(y_true, y_pred_prob, average = 'macro', multi_class = 'ovr'), 5)

          return {'model': model, 
                  'y_pred': y_pred, 
                  'params': {"n_estimator": n_estimators, "criterion": criterion,
                             "max_depth": max_depth, "seed": random_state},
                  'metrics': {"accuracy": accuracy, "precision": precision, "recall": recall, 
                              "roc_score": roc_score}}

def save_model(model: BaseEstimator, model_dir: str, model_name: str = 'model') -> None : 
     try : 
          joblib.dump(model, f'{model_dir}/{model_name}.joblib')
     except Exception as e : 
          infologger.info(f'unable to save the model [check save_model(). exc: {e}')
     else :
          infologger.info(f'model saved successfully [loc: {model_dir}]')

def main() -> None : 
     curr_path = pathlib.Path(__file__) 
     home_dir = curr_path.parent.parent.parent.as_posix()
     params_loc = f'{home_dir}/params.yaml'
     plots_dir = pathlib.Path(f'{home_dir}/figures/training')
     plots_dir.mkdir(parents = True, exist_ok = True)
     try : 
          params = yaml.safe_load(open(params_loc, encoding = 'utf8'))
     except Exception as e :
          infologger.info(f'unable to load params.yaml [check main()]. exc: {e}')
     else : 
          parameters = params['train_model']
          TARGET = params['base']['target']

          train_data = f"{home_dir}{params['build_features']['processed_data']}/processed_train.csv"
          model_dir = f"{home_dir}{parameters['model_dir']}"
          pathlib.Path(model_dir).mkdir(parents = True, exist_ok = True)
          
          data = load_data(train_data)
          X_train = data.drop(columns = [TARGET]).values
          Y = data[TARGET]

          details = train_model(X_train, Y, parameters['n_estimators'], parameters['criterion'], parameters['max_depth'], min_samples_leaf = parameters['min_samples_leaf'],
                                 min_samples_split = parameters['min_samples_split'], random_state = parameters['random_state'], yaml_file_obj = params)

          filename = visualize.conf_matrix(Y, details['y_pred'], labels = details['model'].classes_, path = plots_dir, params_obj = params)

          mlflow_config = params['mlflow_config']
          remote_server_uri = mlflow_config['remote_server_uri']
          exp_name = mlflow_config['trainingExpName']

          mlflow.set_tracking_uri(remote_server_uri)
          mlflow.set_experiment(experiment_name = exp_name)
          # adding experiment description
          experiment_description = ('training RFC model. Obj is to predict the wine quality based on various physicochemical features') 
 
          mlflow.set_experiment_tag("mlflow.note.content", experiment_description)
          
          # runs description
          with mlflow.start_run(description = 'RFC - ronil') : 
               # logging the prarmeters
               mlflow.log_params({"n_estimator": details['params']['n_estimator'], "criterion": details['params']['criterion'], 
                                  "max_depth": details['params']['max_depth'], "seed": details['params']['seed']})
               # logging metrics
               mlflow.log_metrics({"accuracy": details['metrics']['accuracy'], "precision": details['metrics']['precision'], 
                                   "recall": details['metrics']['recall'], "roc_score": details['metrics']['roc_score']})
               # loagging the current run's model
               log_model(details['model'], "model")
               # logging confusion matrix img
               mlflow.log_artifact(filename, 'confusion_matrix')
               # setting tags to each run
               mlflow.set_tags({'project_name': 'wine-quality', 'project_quarter': 'Q1-2024', 'ml_model' : 'RFC'})

          save_model(details['model'], model_dir)

if __name__ == '__main__' : 
     infologger.info('train_model.py as __main__')
     main()
     