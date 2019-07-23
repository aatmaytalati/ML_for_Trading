"""Implement Insane Learner"""
import os
import sys
import numpy as np
import LinRegLearner, DTLearner, RTLearner, BagLearner


class InsaneLearner(object):

    def author(self):
        return 'atalati3' # replace tb34 with your Georgia Tech username
    
    def get_learner_info(self):
        for i in range(1, self.bags + 1):
            self.learners[i-1].get_learner_info() 
    
    def __init__(self, bag_learner=BagLearner.BagLearner, learner=DTLearner.DTLearner, num_bag_learners=20, verbose=False, **kwargs):
        # Initilization of Insane Learner

        # Parameters:
        #       bag_learner: A BagLearner
        #           
        #       learner: A LinRegLearner, DTLearner, or RTLearner to be called by bag_learner
        #       
        #       num_bag_learners: The number of Bag learners to be trained
        # 
        #       verbose: True ->    information about the learner will be printed out
        #                False ->   information about the learner will NOT be printed out
        #       
        #       kwargs: Keyword arguments to be passed on to the learner's constructor.\
        #               **kwargs: The special syntax, *args and **kwargs in function definitions is used to pass a variable number of arguments to a function.
        #               Referance: https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/
        
        # Returns: 
        #       Insane Learner (An instance of)
        
        
        bag_learners = []
        
        for itr in range(num_bag_learners):
            bag_learners.append(bag_learner(learner=learner, **kwargs))
        
        self.bag_learners = bag_learners
        bag_learner = BagLearner.BagLearner
        
        self.kwargs = kwargs
        
        self.num_bag_learners = num_bag_learners
        num_bag_learners = 20
        
        self.verbose = verbose
        verbose = False    
        if verbose:
            self.get_learner_info()

        
    def addEvidence(self, dataX, dataY):
        # feeding_data_to_learn_better

        # Params:
        #   dataX : A numpy ndarray -> x values at each node
        #   dataY : A numpy 1d array -> y values at each node

        # Returns: 
        #   Tree: A mumpy ndarray
        
        if self.verbose:
            self.get_learner_info()  
        
        for bag_learner in self.bag_learners:
            bag_learner.addEvidence(dataX, dataY)
              
    def query(self, points):
        # set of test points given the model we built
        
        # Parameters:
        #     points: A numpy ndarray of test queries

        # Returns: 
        #     preds: A numpy 1D array of estimated values 
        
        predictions_query = np.array([learner.query(points) for learner in self.bag_learners])
        return np.mean(predictions_query, axis=0)


    def get_learner_info(self):
        """Print out data for this InsaneLearner"""
        bag_learner_name = str(type(self.bag_learners[0]))[8:-2]
        print ("This InsaneLearner is made up of {} {}:".
            format(self.num_bag_learners, bag_learner_name))
        print ("kwargs =", self.kwargs)

        # Print out information for each learner within InsaneLearner
        for i in range(1, self.num_bag_learners + 1):
            print (bag_learner_name, "#{}:".format(i)); 
            self.bag_learners[i-1].get_learner_info() 


if __name__=="__main__":
    print ("This is a Insane Learner\n")
    
    # Some data to test the InsaneLearner
    x0 = np.array([0.872, 0.730, 0.575, 0.729, 0.620, 0.265, 0.500, 0.320])
    x1 = np.array([0.325, 0.385, 0.495, 0.569, 0.647, 0.647, 0.689, 0.750])
    x2 = np.array([9.150, 10.890, 9.390, 9.799, 8.392, 11.777, 10.444, 10.001])
    x = np.array([x0, x1, x2]).T
    
    y = np.array([4.000, 5.000, 6.000, 5.000, 3.000, 8.000, 7.000, 6.000])

    # Create an InsaneLearner from given training x and y
    insane_learner = InsaneLearner(verbose=True)
    insane_learner.addEvidence(x, y)