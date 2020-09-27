import numpy as np
import matplotlib.pyplot as plt

def shuffle_data(data):
    t = data['t']
    X = data['X']
    random_permutation = np.random.permutation(len(X))
    res = {'X':[],'t':[]}
    for value in random_permutation:
        res['t'].append(data[0][value])
        res['X'].append(data[1][value])
    return res
    
def split_data(data,num_folds,fold):
    t = data['t']
    X = data['X']
    block_length = len(t) // num_folds
    start_point = block_length * (fold - 1)
    end_point = block_length * fold
    data_fold = {}
    data_rest = {}
    data_fold['X'] = X[start_point:end_point,:]
    data_fold['t'] = t[start_point:end_point]
    data_rest['X'] = np.concatenate((X[:start,:],X[end:,:]))
    data_rest['t'] = np.concatenate((t[:start_point],t[end_point:]))
    return data_fold, data_rest
    
def train_model(data,lambd):
    

def predict(data,model):
    return np.dot(data['X'],model)

def loss(data,model):
    t = data['t']
    prediction = predict(data,model)
    return float(np.sum((t-prediction)**2)/(2*len(t)))

def cross_validation(data,num_folds,lambd_seq):
    cv_error = np.zeros(len(lambd_seq))
    data = shuffle_data(data)
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1,num_folds+1):
            val_cv, train_cv = split_data(data,num_folds,fold)
            model = train_model(train_cv,lambd)
            cv_loss_lmd += loss(val_cv,model)
        cv_error[i] = cv_loss_lmd/num_folds
    return cv_error
    
def train_and_test_error(data_train,data_test,lambd_seq):
    train_error = []
    test_error = []
    for i in lambd_seq:
        model = train_model(data_train,i)
        train_error.append(loss(data_train,model))
        test_error.append(loss(data_test,model))
    return train_error, test_error
    
def draw_graph(train_error,test_error,cv_error_0,cv_error_1,title):

###########------main------###########
if __name__ == '__main__':
    data_train = {'X': np.genfromtxt('data_train_X.csv',delimiter=','),'t': np.genfromtxt('data_train_y.csv',delimiter=',')}
    data_test = {'X': np.genfromtxt('data_test_X.csv',delimiter=','),'t': np.genfromtxt('data_test_y.csv',delimiter=',')}
    lambd_seq = []
    interval = (0.005 - 0.00005) / 49
    for i in range(50):
        lambd_seq.append(0.00005 + interval * i)
        
    train_error, test_error = train_and_test_error(data_train,data_test,lambd_seq)
    
    print(train_error)
    print(test_error)
    
