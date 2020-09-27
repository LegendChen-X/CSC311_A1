import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
def load_data(real: str, fake: str):
    real = open(real, "r")
    fake = open(fake, "r")
    real_lines = real.read().split('\n')
    fake_lines = fake.read().split('\n')
    all_lines = real_lines + fake_lines
    real_labels = [1]*len(real_lines)
    fake_labels = [0]*len(fake_lines)
    all_labels = np.array(real_labels + fake_labels)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_lines)
    train = 0.7
    test = 0.15
    validation = 0.15
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, all_labels, train_size = train)
    X_test, X_validation,Y_test, Y_validation = train_test_split(X_temp,Y_temp, test_size = test/(test+validation))
    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

data_train, data_validation, data_test, label_train, label_validation, label_test = load_data("clean_real.txt", "clean_fake.txt")
def select_knn_model(X_train, X_validation, Y_train, Y_validation):
    train_accuracy = []
    validation_accuracy = []
    for k in range(1,21):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        train_prediction = knn.predict(X_train)
        validation_prediction = knn.predict(X_validation)
        train_accuracy.append(metrics.accuracy_score(Y_train, train_prediction))
        validation_accuracy.append(accuracy_score(Y_validation, validation_prediction))
    k = np.arange(1,21)
    plt.plot(k, validation_accuracy, label = "validation")
    plt.plot(k, train_accuracy, label = "training")
    plt.xticks(np.arange(1, 21, 1.0))
    plt.xlabel("k")
    plt.ylabel('accuracy')
    plt.legend()
    plt.title("Training and Validation accuracy over 20 k's")
    plt.show()
select_knn_model(data_train, data_validation, label_train, label_validation)
