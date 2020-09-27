import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data):
    t = data[0]
    X = data[1]
    random_permutation = np.random.permutation(len(X))
    res = {'X':[],'t':[]}
    for value in random_permutation:
        res['t'].append(data[0][value])
        res['X'].append(data[1][value])
    return res
    
def split_data(data,num_folds,fold):
    

def train_model(data,lambd):

def predict(data,model):

def loss(data,model):

def cross_validation(data,num_folds,lambd_seq):



###########------main------###########
data_train = {'X': np.genfromtxt('data_train_X.csv',delimiter=','),'t': np.genfromtxt('data_train_y.csv',delimiter=',')}
data_test = {'X': np.genfromtxt('data_test_X.csv',delimiter=','),'t': np.genfromtxt('data_test_y.csv',delimiter=',')}
