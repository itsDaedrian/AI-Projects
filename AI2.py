
import numpy as np
from numpy import log, dot, exp, shape
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# to compare our model's accuracy with sklearn model
from sklearn.linear_model import LogisticRegression

def standardize(X_tr):  # (x-Mean(x))/std(X) Normalizes data
    for i in range(shape(X_tr)[1]):
       X_tr[:, i] = (X_tr[:, i] - np.mean(X_tr[:, i])) / np.std(X_tr[:, i])
    return X_tr

def sigmoid(z):#Sigmoid/Logistic function
    sig = 1 / (1 + exp(-z))
    return sig

def cost(theta, X, y):#Cost calculation in Logistic Regression
    z = dot(X, theta)
    cost0 = y.T.dot(log(sigmoid(z)))
    cost1 = (1 - y).T.dot(log(1 - sigmoid(z)))
    cost = -((cost1 + cost0)) / len(y)
  #  print("cost:", cost)
    return cost

def initialize(X):#Initializing X feature matrix and Theta vector
    thetas = np.zeros((shape(X)[1] + 1, 1))
    X = np.c_[np.ones((shape(X)[0], 1)), X]  # adding 691 rows of ones as the first column in X
    return thetas, X

def fit(X, y, alpha=0.001, iter=400):  # Gradient Descent
    thetas, X = initialize(X)
    #making sure that their shapes match
    print("thetas.shape: ", thetas.shape)
    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    cost_list = np.zeros(iter, )
    for i in range(iter):#gradient descent iterations
        #derivative of J(Theta)= (1/m)*(X^T . (h(x)-y))
        thetas = thetas - alpha * dot(X.T, (sigmoid(dot(X, thetas)) - np.reshape(y, (len(y), 1))))#making y a column vector
        cost_list[i] = cost(thetas, X, y)#putting each iteration's cost in a list to display later
    global gthetas#not a good way of programming but works for now:))
    gthetas = thetas
    return cost_list

def predict(X):
    X = np.c_[np.ones((shape(X)[0], 1)), X]  # adding # rows of ones as the first column in X
    z = dot(X, gthetas)
    sig = sigmoid(z)
    lis = []
    for i in sig:
        if i > 0.5:#notice that boundary included in class 0
            lis.append(1)
        else:
            lis.append(0)
    # Another, more elegant way is: lis = np.where( sig >= 0.5, 1, 0 ) in this case boundary included in class 1
    return lis

def compare(Y_test, Y_pred, Y_pred1):
    # measure performance
    correctly_classified = 0
    correctly_classified1 = 0
    # counter
    count = 0
    for count in range(np.size(Y_pred)):
        if Y_test[count] == Y_pred[count]:
            correctly_classified = correctly_classified + 1
        if Y_test[count] == Y_pred1[count]:
            correctly_classified1 = correctly_classified1 + 1
        count = count + 1
    print("Accuracy on test set by our model       :  ", (
            correctly_classified / count) * 100)
    print("Accuracy on test set by sklearn model   :  ", (
            correctly_classified1 / count) * 100)

def main():
    data = pd.read_csv('Social_Network_Ads.csv')
    new_df = data
    new_df.replace('Female', int(0), inplace=True)
    new_df.replace('Male', int(1), inplace=True)
    X = data.iloc[:, :-1].values  # it means you are taking all the rows and all the columns except the last column which is Y
    y = data.iloc[:, -1:].values  # take all rows in the last column

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    X_tr = standardize(X_tr)
    cost_list = fit(X_tr, y_tr)
    plt.scatter(range(len(cost_list)), cost_list, c="blue")
    plt.show()

    y_pred = predict(X_te)#predicting with our own Logistic Regression implementation
    #Predicting with scikit learn's LogisticRegression() implementation
    model1 = LogisticRegression()
    model1.fit(X_tr, y_tr)
    y_pred1 = model1.predict(X_te)

    compare(y_te, y_pred, y_pred1)

main()#main driving function call
