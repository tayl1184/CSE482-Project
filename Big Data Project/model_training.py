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
    
    
    
    #linear regression classifier
    clf = linear_model.LogisticRegression(C=10, random_state=1)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    logistic_acc = accuracy_score(Y_test, Y_pred)
    
    
    
    
    
    
    #neural network classifier
    clf = neural_network.MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', 
                                  max_iter=2000, random_state=1 , early_stopping=True)
    clf = clf.fit(X_train, Y_train.ravel())
    Y_pred = clf.predict(X_test)
    nn_acc = accuracy_score(Y_test, Y_pred)
    
    
    
    
    
    #Compare accuracy of best depth decision tree and log regression
    methods = ['Dtree', 'LogRegression', 'Neural Network']
    acc = [testAcc[bestHyperparam], logistic_acc, nn_acc]
    plt.title(f'Decision Tree vs Logistic Regression vs NN for {data_order[i]}')
    plt.bar([1.5,2.5, 3.5],acc)
    plt.xticks([1.5,2.5, 3.5], methods)
    plt.ylabel('Test accuracy')
    plt.ylim([0.0,1])
    plt.show()
    print(f"Best classification method for {data_order[i]}: {methods[acc.index(max(acc))]}")
     
    """
    #results of decision trees
    
    
    plt.plot(maxdepths, validationAcc, 'ro--', maxdepths, testAcc, 'kv-', maxdepths)
    plt.xlabel('Maximum depth')
    plt.ylabel('Accuracy')
    plt.title(f'{data_order[i]}')
    plt.legend(['Validation','Testing'])
    plt.ylim([0.1,0.65])
    plt.show()
   
    
    print(f"Test accuracy for {data_order[i]}: {testAcc[bestHyperparam]}" )
    print(f"Best hyperparameter for {data_order[i]} , MaxDepth = {maxdepths[bestHyperparam]}")
print()
print(f"Average Test Accuracy for all trees: {sum(best_test_accuracies) / len(best_test_accuracies) }")
    
 """