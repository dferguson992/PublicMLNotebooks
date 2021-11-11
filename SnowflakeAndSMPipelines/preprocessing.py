
import glob
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

if __name__=='__main__':
    
    input_files = glob.glob('{}/*.npy'.format('/opt/ml/processing/input'))
    print('\nINPUT FILE LIST: \n{}\n'.format(input_files))
    scaler = StandardScaler()
    
    for file in input_files:
        if 'x_' in file:
            raw_x = np.load(file)
        if 'y_' in file:
            raw_y = np.load(file)
        else:
            raw_y = None
            
    transformed = scaler.fit_transform(raw_x, raw_y)
    for file in input_files:
        raw = np.load(file)
        transformed = scaler.fit_transform(raw)
        if 'train' in file:
            output_path = os.path.join('/opt/ml/processing/train', 'xtrain.npy')
            np.save(output_path, transformed)
            print('SAVED TRANSFORMED TRAINING DATA FILE\n')
        else:
            output_path = os.path.join('/opt/ml/processing/test', 'xtest.npy')
            np.save(output_path, transformed)
            print('SAVED TRANSFORMED TEST DATA FILE\n')
