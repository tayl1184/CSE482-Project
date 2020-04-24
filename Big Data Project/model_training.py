import pandas as pd

import warnings
warnings.filterwarnings("ignore")


top_followed = pd.read_csv("top_followed.csv") # source: https://socialblade.com/twitter/top/500/followers
headers = [top_followed['screen_name'][x] for x in range(20)]
headers.insert(0, 'sentiment')



CNN_class_data = pd.read_csv("CNN_class_data.csv", names=headers)
FoxNews_class_data = pd.read_csv("FoxNews_class_data.csv", names=headers)
MSNBC_class_data = pd.read_csv("MSNBC_class_data.csv", names=headers)
NPR_class_data = pd.read_csv("NPR_class_data.csv", names=headers)
cspan_class_data = pd.read_csv("cspan_class_data.csv", names=headers)



sentiment_data = [CNN_class_data, FoxNews_class_data, MSNBC_class_data, NPR_class_data, cspan_class_data]
data_order = ["CNN" , "FOX", "MSNBC" , "NPR" , "CSPAN"]

#training model
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import neural_network


best_test_accuracies = []
for i, news_outlet in enumerate(sentiment_data):
    Y = pd.Series(news_outlet['sentiment'].values)
    X = news_outlet.drop(labels = 'sentiment' , axis = 1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3 , train_size = .7 , random_state=1)
    
    #Decision Tree Classifier
    
    clf = tree.DecisionTreeClassifier();
    clf = clf.fit(X_train, Y_train)
    maxdepths = [1,2,3,4,5,6,7,8,9,10]
    validationAcc = np.zeros(len(maxdepths))
    testAcc = np.zeros(len(maxdepths))

    numFolds = 10
    
    #Finding the best hyperparams and calculating error of trees using cross validation
    index = 0
    
    for depth in maxdepths:
        clf = tree.DecisionTreeClassifier(max_depth = depth, random_state = 1)
        scores = cross_val_score(clf, X_train, Y_train, cv= numFolds)
        validationAcc[index] = np.mean(scores)
     
        clf = clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        
        testAcc[index] = accuracy_score(Y_test, Y_pred)
        
        bestHyperparam = np.argmax(validationAcc)
        index += 1
        
    best_test_accuracies.append(testAcc[bestHyperparam])


    params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    logistic_acc = np.zeros(len(params))
    coef_log = []
    index = 0
    #linear regression classifier
    for value in params:
        clf = linear_model.LogisticRegression(C=value, random_state=1)
        
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        
        logistic_acc[index] = (accuracy_score(Y_test, Y_pred))
     #   print("param : " , value , clf.coef_)
        coef_log.append(clf.coef_)
        best_log_param = np.argmax(logistic_acc)

        index += 1
    best_coef_logs = coef_log[best_log_param]

    #neural network classifier
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', 
                                  max_iter=2000, random_state=1 , early_stopping=True)
    clf = clf.fit(X_train, Y_train.ravel())
    Y_pred = clf.predict(X_test)
    nn_acc = accuracy_score(Y_test, Y_pred)
      

    
    #Compare accuracy of prediction models
    methods = ['Dtree', 'LogRegression', 'Neural Network']
    acc = [testAcc[bestHyperparam], logistic_acc[best_log_param], nn_acc]
    plt.title(f'Decision Tree vs Logistic Regression vs NN for {data_order[i]}')
    plt.bar([1.5,2.5, 3.5],acc)
    plt.xticks([1.5,2.5, 3.5], methods)
    plt.ylabel('Test accuracy')
    plt.ylim([0.0,1])
    plt.show()
    print(f"The Best classification method for {data_order[i]} is {methods[acc.index(max(acc))]} with accuracy : {round(acc[acc.index(max(acc))] , 3)}")
    print(f"Best Hyperparams DTree : {maxdepths[bestHyperparam]}  LogReg : {params[best_log_param]} ")
    print("Log coefficients : " , best_coef_logs)
    
