from math import log2
import numpy as np 
import pandas as pd 
def loss(y, yhat, m):
    y_zero_loss = y * np.log(yhat + 1e-9)
    y_one_loss = (1-y) * np.log(1 - yhat + 1e-9)
    J = -np.mean(y_zero_loss + y_one_loss)/m 
    return J 

def initialize(dim):
    theta = np.random.rand(dim)
    b = np.random.random()
    return theta, b 

def update_theta(X, y, yhat, lr, theta, b):
    db=(np.sum(yhat-y)*2)/len(y)
    dw=(np.dot(2*(yhat-y),X))/len(y)
    b_1=b-lr*db
    theta_1=theta-lr*dw
    
    return b_1,theta_1
    
def predict(theta, X):
    return 1/(1 + np.exp(-np.dot(X, theta)))
    

def run_gradient_decent(X, y, alpha, iterations):
    theta, b = initialize(X.shape[1])
    gd_iterations_df=pd.DataFrame(columns=['iteration','cost'])
    iter_num = result_index = 0
    for iter in range(iterations):
        yhat = predict(theta, X)
        cost = loss(y, yhat, X.shape[0])
        b, theta = update_theta(X, y, yhat, alpha, theta, b)
        if iter % 10 == 0:
            gd_iterations_df.loc[result_index] = [iter_num, cost]
        result_index += 1
        iter_num += 1
    return gd_iterations_df, b, theta
def main():
    X = np.array([[1, 2], [6, 7], [9, 7], [1, 5], [8, 4]])
    y = np.array([1, 1, 1, 0, 0])
    gd_iterations_df, b, theta = run_gradient_decent(X, y, alpha = 0.01, iterations=100)
    print(gd_iterations_df)
    print(predict(theta, X))
main()
    
    
    





