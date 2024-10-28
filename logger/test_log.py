from datetime import datetime
import pickle
import os

def save_test_log(test_values, name):
    file_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
    dir_path = f'log/test/{name}'
    file_path = f'{dir_path}/{file_name}.pkl'
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    with open(file_path, 'wb') as f:
        pickle.dump(test_values, f)
    
    print(f'Test log saved at {file_path}')
    return file_path

def load_test_log(file_path):
    with open(file_path, 'rb') as f:
        values = pickle.load(f)
    return values

dummy_test_values = [i*i for i in range(-100, 100)]
dummy_test_name = 'dummy'

if __name__ == '__main__':
    save_test = save_test_log(dummy_test_values, dummy_test_name)
    print(load_test_log(save_test))