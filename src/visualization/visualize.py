import pathlib
import yaml
import typing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.logger import infologger
from sklearn.base import BaseEstimator

infologger.info('*** Executing: visualize.py ***')
from src.data.make_dataset import load_data

def load_model(model_dir: str) -> BaseEstimator :
     try : 
          model = joblib.load(model_dir)
     except Exception as e : 
          infologger.info(f'unable to load the model from {model_dir} [check load_model()]. exc: {e}')
     else : 
          infologger.info(f'model loaded successfully [loc: {model_dir}]')
          return model

def roc_curve() -> None : 
     pass

def conf_matrix(y_test: pd.Series, y_pred: pd.Series, labels: np.ndarray, path: pathlib.Path, params_obj: typing.IO) -> str : 
     try : 
          curr_time = datetime.now().strftime('%d%m%y-%H%M%S')
          dir_path = pathlib.Path(f'{path}/cMatrix')
          dir_path.mkdir(parents = True, exist_ok = True)
     except Exception as e : 
          infologger.info(f'something wrong with directories [check conf_matrix()]. exc: {e}')
     else :
          infologger.info('directories are all set!')
          try :
               cm = confusion_matrix(y_test, y_pred, labels = labels)
               disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
          except Exception as e : 
               infologger.info(f'unable to plot the confusion metrix [check conf_matrix()]. exc: {e}')
          else :
               disp.plot(cmap = plt.cm.Blues)
               plt.title('Confusion Matrix')
               plt.xlabel('Predicted Label')
               plt.ylabel('True Label')
               filename = f'{dir_path.as_posix()}/{curr_time}.png'
               plt.savefig(filename)
               plt.close()
               infologger.info(f'confusion metrix saved at [{dir_path}]')
               return filename
          
def main() -> None :
     curr_dir = pathlib.Path(__file__)
     home_dir = curr_dir.parent.parent.parent.as_posix()
     dir_path = f'{home_dir}/figures'
     dir_path.mkdir(parents = True, exist_ok = True)
     params = yaml.safe_load(open(f'{home_dir}/params.yaml'))
     data_dir = f"{home_dir}{params['build_features']['processed_data']}/processed_test.csv"
     model_dir = f'{home_dir}{params["train_model"]["model_dir"]}/model.joblib'
     
     TARGET = params['base']['target']

     test_data = load_data(data_dir)
     x_test = test_data.drop(columns = [TARGET]).values
     y_test = test_data[TARGET]
     
     model = load_model(model_dir)
     labels = model.classes_
     try : 
          y_pred = model.predict(x_test)     # return class
     except Exception as e : 
          infologger.info(f'unable to make prediction [check main()]. exc: {e}')
     else :
          conf_matrix(y_test, y_pred, labels, dir_path, yaml_file_obj = params)
               
if __name__ == '__main__' :
     infologger.info('visualize.py as __main__')
     main()
