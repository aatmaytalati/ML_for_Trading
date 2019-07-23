# Bag Learner : Implements Boosting. 
# Aatmay S. Talati
# atalati3

import sys
import os
import numpy as np
import LinRegLearner, DTLearner, RTLearner


class BagLearner(object):

    def author(self):
        return 'atalati3' # replace tb34 with your Georgia Tech username
    
    def get_learner_info(self):
        for i in range(1, self.bags + 1):
            self.learners[i-1].get_learner_info() 
    
    def __init__(self, learner, bags=20, boost=False, verbose=False, **kwargs):
        # Intilization of a Bag Learner
        
        # Parameters:
        #       learner: A Linear Regression learner, Decision Tree learner and random tree learner. 
        #       bags: The number of learners to be trained using Bootstrap Aggregation
        #              We are considering 20 in this assignment. 
        #       boost: True -> boosting will be implemented
        #               False -> Boosting will not be implemented. 
        #        verbose: True ->    information about the learner will be printed out
        #                False ->   information about the learner will NOT be printed out
        #       
        #       kwargs: Keyword arguments to be passed on to the learner's constructor.\
        #               **kwargs: The special syntax, *args and **kwargs in function definitions is used to pass a variable number of arguments to a function.
        #               Referance: https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/
        
        # Returns:
        #         Bag Learner (An Instance)

        
        learners = []
        for i in range(bags):
            learners.append(learner(**kwargs))
        
        self.learners = learners
        
        self.kwargs = kwargs
        
        self.bags = bags
        bags = 20

        self.boost = boost
        boost = False
        
        self.verbose = verbose
        if verbose:
            self.get_learner_info()

        
    def addEvidence(self, dataX, dataY):
        # feeding_data_to_learn_better

        # Params:
        #   dataX : A numpy ndarray -> x values at each node
        #   dataY : A numpy 1d array -> y values at each node

        # Returns: 
        #   Tree: A mumpy ndarray with Updated elements. 
    
        # Sample the data with replacement
        if self.verbose:
            self.get_learner_info()

        for learner in self.learners:
            i_temp = np.random.choice(dataX.shape[0], dataX.shape[0])
            learner.addEvidence(dataX[i_temp], dataY[i_temp])
        
    def query(self, points):
        # COmments same as DTLearner and
        prediction_temp = np.array([learner.query(points) for learner in self.learners])
        return np.mean(prediction_temp, axis=0)

if __name__=="__main__":
    print ("This is a Bag Learner\n")

    # Some data to test the BagLearner
    x0 = np.array([0.872, 0.730, 0.575, 0.729, 0.620, 0.265, 0.500, 0.320])
    x1 = np.array([0.325, 0.385, 0.495, 0.569, 0.647, 0.647, 0.689, 0.750])
    x2 = np.array([9.150, 10.890, 9.390, 9.799, 8.392, 11.777, 10.444, 10.001])
    x = np.array([x0, x1, x2]).T
    
    y = np.array([4.000, 5.000, 6.000, 5.000, 3.000, 8.000, 7.000, 6.000])

    # Create a BagLearner from given training x and y
    bag_learner = BagLearner(DTLearner.DTLearner, verbose=True)
    bag_learner.addEvidence(x, y)
