import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

###########------functions------###########
def load_data(clean_real_path: str,clean_fake_path: str):
# Get the file pointer
    real_news_file = open(clean_real_path,'r')
    fake_news_file = open(clean_fake_path,'r')
# Load news into list
    real_news = [line.rstrip('\n') for line in real_news_file]
    fake_news = [line.rstrip('\n') for line in fake_news_file]
# Combine real & fake news
    all_news = list(real_news+fake_news)
# Get the total length
    real_news_num = len(real_news)
    fake_news_num = len(fake_news)

    total_num = real_news_num + fake_news_num
# Get Cv
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_news)
# Make labels
    real_labels = np.ones((real_news_num,1))
    fake_labels = np.zeros((fake_news_num,1))
    
    labels = np.concatenate((real_labels, fake_labels))
# Split them 7:3
    X_train, X_test, y_train, y_test  = train_test_split(X,labels,train_size=0.7)
# Split them 5 : 5 (Total is 0.3, both of them 0.15)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test,y_test,train_size=0.5)
    
    return {"X_train": X_train, "X_valid": X_valid, "X_test": X_test, "y_train": y_train, "y_valid": y_valid, "y_test": y_test}
    
def select_knn_model(clean_real_path: str,clean_fake_path: str,improvement: bool):
    result = {}
    data_set = load_data(clean_real_path,clean_fake_path)
    #Get data
    X_train = data_set["X_train"]
    X_valid = data_set["X_valid"]
    X_test = data_set["X_test"]
    #Get label
    y_train = data_set["y_train"]
    y_valid = data_set["y_valid"]
    y_test = data_set["y_test"]
    
    best_k = -1
    best_perform = 0.0
    
    k_set = [(i+1) for i in range(20)]
    for value in k_set:
        if improvement:
            model = KNeighborsClassifier(n_neighbors = value,metric = 'cosine')
        else:
            model = KNeighborsClassifier(n_neighbors = value)
        model.fit(X_train,y_train.ravel())
        train_predicted = model.predict(X_train)
        valid_predicted = model.predict(X_valid)
# Check train set & get accuracy
        train_count = 0
        for i in range(len(train_predicted)):
            if train_predicted[i] == y_train[i]:
                train_count += 1
        result[value]=[train_count/len(train_predicted)]
# Check validate set & get accuracy
        valid_count = 0
        for i in range(len(valid_predicted)):
            if valid_predicted[i] == y_valid[i]:
                valid_count += 1
                
        result[value].append(valid_count/len(valid_predicted))
        
        if(result[value][-1]>best_perform):
            best_k = value
            best_perform = result[value][-1]
            
# Check test set with best-performed k
    if improvement:
        model = KNeighborsClassifier(n_neighbors = best_k,metric = 'cosine')
    else:
        model = KNeighborsClassifier(n_neighbors = best_k)
    model.fit(X_train,y_train.ravel())
    test_predicted = model.predict(X_test)
    test_count = 0
    for i in range(len(test_predicted)):
        if test_predicted[i] == y_test[i]:
            test_count += 1
            
    result[0]=[test_count/len(test_predicted)]
    print("Best performance K is",best_k)
    print("Perfoemance with best K is",result[0])
    return result
    
def draw_graph(res,title):
    train_set = [res[item][0] for item in range(1,21)]
    validation_set = [res[item][-1] for item in range(1,21)]
    k_set = [(i+1) for i in range(20)]

    plt.plot(k_set,train_set,label="training")
    plt.plot(k_set,validation_set,label="validation")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.xticks(k_set)
    plt.title(title)
    plt.legend()
    plt.show()
    
###########------main------###########
if __name__ == '__main__':
    res = select_knn_model("clean_real.txt","clean_fake.txt",False)
    draw_graph(res,"Original Graph")
    for i in range(1,21):
        print("When k is",i,", the training accuracy is",res[i][0],", the validation accuracy is",res[i][1])
    improve_res = select_knn_model("clean_real.txt","clean_fake.txt",True)
    draw_graph(improve_res,"Improvement Graph")
