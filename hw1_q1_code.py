import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def load_data(clean_real_path,clean_fake_path):
    real_news_file = open(clean_real_path,'r')
    fake_nes_file = open(clean_fake_path,'r')
    
    real_news = [line.rstrip('\n') for line in real_news_file]
    fake_news = [line.rstrip('\n') for line in fake_nes_file]
    
    all_news = list(real_news+fake_news)
    
    real_news_num = len(real_news)
    fake_news_num = len(fake_news)
    
    total_num = real_news_num + fake_news_num
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_news).toarray()
    
    X_feature = vectorizer.get_feature_names()
    
    real_labels = np.ones((real_news_num,1))
    fake_labels = np.zeros((fake_news_num,1))
    
    labels = np.concatenate((real_labels, fake_labels))
    
    X_train, X_test, y_train, y_test  = train_test_split(X,labels,test_size=0.7)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test,y_test,test_size=0.5)
    
    return {"X_train": X_train, "X_valid": X_valid, "X_test": X_test, "y_train": y_train, "y_valid": y_valid, "y_test": y_test, "X_feature": X_feature}
    
    
def select_knn_model(clean_real_path,clean_fake_path):
    result = {}
    data_set = load_data(clean_real_path,clean_fake_path)
    #Get data
    X_train = data_set["X_train"]
    X_valid = data_set["X_valid"]
    #Get label
    y_train = data_set["y_train"]
    y_valid = data_set["y_valid"]
    
    k_set = [(i+1) for i in range(20)]
    for value in k_set:
        model = KNeighborsClassifier(n_neighbors = value)
        model.fit(X_train,y_train.ravel())
        train_predicted = model.predict(X_train)
        valid_predicted = model.predict(X_valid)
        
        train_count = 0
        for i in range(len(train_predicted)):
            if train_predicted[i] != y_train[i]:
                train_count += 1
        result[value]=[train_count/len(train_predicted)]
        
        valid_count = 0
        for i in range(len(valid_predicted)):
            if valid_predicted[i] != y_valid[i]:
                valid_count += 1
                
        result[value].append(valid_count/len(valid_predicted))
        
    return result
                
        
print(select_knn_model("clean_real.txt","clean_fake.txt"))
    
    
    
    
    
