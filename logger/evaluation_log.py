from datetime import datetime
import pickle
import os

def save_eval_log(eval_values, name):
    file_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
    dir_path = f'log/eval/{name}'
    file_path = f'{dir_path}/{file_name}.pkl'
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    with open(file_path, 'wb') as f:
        pickle.dump(eval_values, f)
    
    print(f'Evaluation log saved at {file_path}')
    return file_path

def load_eval_log(file_path):
    with open(file_path, 'rb') as f:
        eval_values = pickle.load(f)
    return eval_values

dummy_eval_values = ([i*i for i in range(-100, 100)], 'dummy')

if __name__ == '__main__':
    save_test = save_eval_log(dummy_eval_values)
    print(load_eval_log(save_test))