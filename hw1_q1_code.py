import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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
    
    label = np.vstack((np.zeros((fake_news_num, 1)), np.ones((real_news_num, 1))))
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,train_target,test_size=0.7)
    
    
    
load_data("clean_real.txt","clean_fake.txt")
    
