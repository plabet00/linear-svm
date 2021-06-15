import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.utils import shuffle

def removeLessSignificantFeatures(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped

def costCal(W,X,y):
    weight = 10000
    N = X.shape[0]
    cost = 1 - np.dot(X, W) * y
    cost = np.where(cost < 0, 0, cost)
    hinge_loss = weight * (np.sum(cost)/N)

    cost = hinge_loss
    return cost

def stohasticGradientDescent(W, X, y):
    weight = 10000
    X = np.array([X])
    y = np.array([y])
    loss = 1 - np.dot(X, W) * y
    dw = np.zeros(len(W))
    if max(0, loss) == 0:
        dw = W
    else:
        dw = W - (weight * y[0] * X[0])
    return dw

def train(X_train,y_train):
    no_of_iterations = 5000
    nth = 0
    prev_loss = float("inf")
    epsilon = 0.01
    learning_rate = 0.00001
    W = np.zeros(X_train.shape[1])
    for iteration in range(1,no_of_iterations):
        X,y = shuffle(X_train,y_train)
        for i, x in enumerate(X):
            gradient = stohasticGradientDescent(W,x,y[i])
            W = W - (learning_rate * gradient)
        if iteration == 2 ** nth or iteration == no_of_iterations - 1:
            loss = costCal(W, X_train, y_train)
            if abs(prev_loss - loss) < epsilon * prev_loss:
                return W
            prev_loss = loss
            nth += 1

    return W

def testModel(model, X_test, y_test):
    correct = 0
    for x in range(len(y_test)):
        output = y_test[x] * np.dot(X_test[x],model)
        if output > 0:
            correct += 1
    print('Accuracy:',correct/len(y_test))

data = pd.read_csv('./heart.csv')
y = data.loc[:,'target']
for x in range(len(y)):
  if y[x] == 0:
    y[x] = -1
X = data.iloc[:, :-1]
removeLessSignificantFeatures(X,y)

X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)
X.insert(loc=len(X.columns), column='intercept', value=1)

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=42)

model = train(X_train.to_numpy(),np.array(y_train))
testModel(model, X_test.to_numpy(),np.array(y_test))
