import pathlib
import yaml
import numpy as np
import pandas as pd
from src.logger import infologger

infologger.info("*** Executing: build_features.py ***")
# writing import after infologger to log the info precisely 
from src.data.make_dataset import load_data

# build augmented features
def feat_eng(df: pd.DataFrame, name: str = 'default') -> pd.DataFrame :
     try : 
          df.columns = df.columns.str.replace(' ', '_')
          df['total_acidity'] = df['fixed_acidity'] + df['volatile_acidity'] + df['citric_acid']
          df['acidity_to_pH_ratio'] = df['total_acidity'] / df['pH']
          df['free_sulfur_dioxide_to_total_sulfur_dioxide_ratio'] = df['free_sulfur_dioxide'] / df['total_sulfur_dioxide']
          df['alcohol_to_acidity_ratio'] = df['alcohol'] / df['total_acidity']
          df['residual_sugar_to_citric_acid_ratio'] = df['residual_sugar'] / df['citric_acid']
          df['alcohol_to_density_ratio'] = df['alcohol'] / df['density']
          df['total_alkalinity'] = df['pH'] + df['alcohol']
          df['total_minerals'] = df['chlorides'] + df['sulphates'] + df['residual_sugar']
          
          # Cleaning inf or null values that may result from the operations above
          df = df.replace([np.inf, -np.inf], 0)
          df = df.dropna()

          df[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
              'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
              'pH', 'sulphates', 'alcohol', 'total_acidity', 'acidity_to_pH_ratio', 
              'free_sulfur_dioxide_to_total_sulfur_dioxide_ratio',
              'alcohol_to_acidity_ratio', 'residual_sugar_to_citric_acid_ratio',
              'alcohol_to_density_ratio', 'total_alkalinity', 'total_minerals']] = df[['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                                                                                       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                                                                                       'pH', 'sulphates', 'alcohol', 'total_acidity', 'acidity_to_pH_ratio', 
                                                                                       'free_sulfur_dioxide_to_total_sulfur_dioxide_ratio',
                                                                                       'alcohol_to_acidity_ratio', 'residual_sugar_to_citric_acid_ratio',
                                                                                       'alcohol_to_density_ratio', 'total_alkalinity', 'total_minerals']].round(4)
     except Exception as e : 
          infologger.info(f'unable to perform feature eng [check feat_eng()]. exc: {e}')
     else :
          infologger.info(f'features generated successfully!')
          return df

# save the data
def save_data(path: str, train: pd.DataFrame, test: pd.DataFrame) -> None : 
     try : 
          train.to_csv(f'{path}/processed_train.csv', index = False)
          test.to_csv(f'{path}/processed_test.csv', index = False)
     except Exception as e : 
          infologger.info(f'unable to save the data [check save_data()]. exc: {e}')
     else : 
          infologger.info(f'features generated successfully for training & testing data [loc: {path}]')

def main() -> None : 
     curr_dir = pathlib.Path(__file__)
     home_dir = curr_dir.parent.parent.parent.as_posix()
     
     params_file_loc = f'{home_dir}/params.yaml'
     try : 
          params = yaml.safe_load(open(params_file_loc))
     except Exception as e : 
          infologger.info(f'unable to load params.yaml [check main()]. exc: {e}')
     else :
          parameters = params['build_features']
          processed_data = params['make_dataset']['processed_data']
          
          # load the data
          train_data_loc = f'{home_dir}{processed_data}/train.csv'
          test_data_loc = f'{home_dir}{processed_data}/test.csv'
          train_data, test_data = load_data(train_data_loc), load_data(test_data_loc)
          # save the data 
          path = f'{home_dir}{parameters["processed_data"]}'
          save_data(path, feat_eng(train_data, 'training_data'), feat_eng(test_data, 'testing_data'))

if __name__ == "__main__" : 
     infologger.info('build_features as __main__')
     main()
     