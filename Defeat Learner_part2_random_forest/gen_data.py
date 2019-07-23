"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""
import os
import sys
import math
import numpy as np

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees

def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    # Here's is an example of creating a Y from randomly generated
    # X with multiple columns
    # Y = X[:,0] + np.sin(X[:,1]) + X[:,2]**2 + X[:,3]**3

    # X and Y should each contain from 10 to 1000 rows
    num_rows = np.random.randint(10, 1001)

    # X should have from 2 to 1000 columns
    num_X_cols = np.random.randint(2, 1001)

    X = np.random.normal(size=(num_rows, num_X_cols))
    Y = np.zeros(num_rows)
    for col in range(num_X_cols):
        Y += X[:, col]

    return X, Y


def best4DT(seed=1489683273):

    np.random.seed(seed)
    
    # X and Y should each contain from 10 to 1000 rows
    num_rows = np.random.randint(10, 1001)

    # X should have from 2 to 1000 columns
    num_X_cols = np.random.randint(2, 1001)

    X = np.random.normal(size=(num_rows, num_X_cols))
    Y = np.zeros(num_rows)
    for col in range(num_X_cols):
        Y += (X[:, col] * X[:, col])
    
    return X, Y

def author():
    return 'atalati3' #Change this to your user ID

if __name__=="__main__":
    print "they call me Tim."