# Decision Tree Learner
# Aatmay S. Talati
# atalati3

#Import Statments 
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr as pr
from collections import Counter as cntr

# Primary Resources of DTLearner:
#   1. https://www.youtube.com/watch?v=OBWL4oLT7Uc
#   2. https://www.youtube.com/watch?v=WVc3cjvDHhw
#   3. https://github.com/ntrang086/ml_trading_assess_learners

class DTLearner(object):
    
    def author(self):
        return 'atalati3' # replace tb34 with your Georgia Tech username        
    
    def get_learner_info(self):
        for i in range(1, self.bags + 1):
            self.learners[i-1].get_learner_info() 
    
    def __init__(self, leaf_size=1, verbose=False, tree=None, **kwargs):
        # Treee initilization, and assigning values. 
    
        # Parameters:
        #           leaf_size:  The maximum number of samples to be aggregated at a leaf
        #           verbose: True ->    information about the learner will be printed out
        #                    False ->   information about the learner will NOT be printed out
        #           tree: If None -->       the learner instance has no data. 
        #                 If not None ->    tree is a numpy ndarray. 
        #           **kwargs: The special syntax, *args and **kwargs in function definitions is used to pass a variable number of arguments to a function.
        #           Referance: https://www.saltycrane.com/blog/2008/01/how-to-use-args-and-kwargs-in-python/
        #
        #                <------ feature indices (int type; index for a leaf is -1), splitting values ------>
        #                           /     ~            ~               ~              ~
        #                           |     ~            ~               ~              ~
        #               nodes       |     ~            ~               ~              ~
        #                           |     ~            ~               ~              ~
        #                           \     ~            ~               ~              ~
        #
        #           Its columns are the features of data and its rows are the individual samples. The four 
        #           columns are feature indices (index for a leaf is -1), splitting values (or Y values for
        #           leaves), and starting rows, from the current root, for its left and right subtrees (if any)
        
        # Returns: 
        #       Decision Tree Learner (An instance)
        
        self.leaf_size = leaf_size
        leaf_size == 1

        self.verbose = verbose
        verbose == False

        self.tree = tree
        tree == None

        if verbose:
            self.get_learner_info()
        
        self.kwargs = kwargs

    def split_on_best_feature(self, dataX, dataY):
        # Resource: Primary Resource -> No. 2
        # Main Goal: Choosing the best feature to split on <- that means choosing the feature which has the highest absoluate correlation with dataY.
        # Rules:
        #       1. Splitting value will be the mean of the splitiing feature values. 
        #           1a. If all the features have same amount of values, then choose the feature which comes first. 
        #       2. If the selected best feature can not split the data, then we will choose second best feature to split on. 
        #           2a. If none of the feature can not split the data accordingly, then it that case we will return the leaf. 

        # Params:
        #   dataX : A numpy ndarray -> x values at each node
        #   dataY : A numpy 1d array -> y values at each node

        # Returns: 
        #   Tree: A mumpy ndarray.
        #       
        #                <------ feature indices (int type; index for a leaf is -1), splitting values ------>
        #                           /     ~            ~               ~              ~
        #                           |     ~            ~               ~              ~
        #               nodes       |     ~            ~               ~              ~
        #                           |     ~            ~               ~              ~
        #                           \     ~            ~               ~              ~

        if dataX.shape[0] <= self.leaf_size: 
            return np.array([-1, cntr(dataY).most_common(1)[0][0], np.nan, np.nan])
    
        # Now, lets look into the availble list of features. 
        availble_features = range(dataX.shape[1]) # Equivalent to num_features
        availble_LIST_of_features = list(availble_features)

        # Tuples: (<features>, <their_correlation_with_dataY>)
        feature_correlations = []
        feature_correlations = sorted(feature_correlations, key=lambda feature_correlations: feature_correlations[1]) # Sorting with correlations.
        # Referance for Sorting: https://docs.python.org/2.7/howto/sorting.html
         
        for ftr_itr in range(dataX.shape[1]):
            absolute_correlation_value = abs(pr(dataX[:, ftr_itr], dataY)[0])
            
            # Dropping NAN values, and assigning their correlation to 0.0 <- float number.
            if np.isnan(absolute_correlation_value):
                absolute_correlation_value = 0.0
            else:
                pass            
            
            # Now,Appending all values to features_correlaticat coons.            
            feature_correlations.append((ftr_itr, absolute_correlation_value))
        
        # Choosing the best feature. 
        # if lenth of availble total features are 0,
        #           then return leaf. 
        feature_Correlation_temp = 0   
        if len(availble_LIST_of_features) == 0:
            return np.array([-1, cntr(dataY).most_common(1)[0][0], np.nan, np.nan])
        
        #else:
        #   once again check if the features are 1 or more. 
        # Choose the best feature, if any, by iterating over feats_corrs
        else:            
            # Choose the best feature, if any, by iterating over feats_corrs
            while len(availble_LIST_of_features) -1 >= 0:
                best_feature_itr = feature_correlations[feature_Correlation_temp][0]
                y = best_feature_itr

                # Split the data according to the best feature, and considering the mean of the data. 
                # Primary Resource No. 2
                split_val = np.median(dataX[:, y])

                # Arrays for indexing - Logically 
                left_i = dataX[:, y] 
                right_i = dataX[:, y]

                left_index = left_i <= split_val
                right_index = right_i > split_val

                # In any case if we can not split ANY feature in any two distinct parts, then all we do is -> return the leaf.         
                if len(np.unique(left_index)) != 1:
                    break                
                # Once we use the feature, then we take it off from remaining best features to choose from. 
                availble_LIST_of_features.remove(y)
                feature_Correlation_temp = feature_Correlation_temp + 1            
                      
        # Once we run while loop and in any case if we run out of all features that we can split on, then in that case we just return leaf. 
        if len(availble_LIST_of_features) == 0:
            return np.array([-1, cntr(dataY).most_common(1)[0][0], np.nan, np.nan])      
        
        # Building Following:
        #       left branch
        #       the root                    
        lefttree = (self.split_on_best_feature(dataX[left_index], dataY[left_index]))

        # Set the starting row for the right subtree of the current root
        if lefttree.ndim == 1: # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.ndim.html
            righttree_start = 1 
            righttree_start = righttree_start + 1
        elif lefttree.ndim >= (1+1):
            righttree_start = (lefttree.shape[0] + 2)-1
        root = np.array([best_feature_itr, split_val, 1, righttree_start])

        return np.vstack((root, lefttree, self.split_on_best_feature(dataX[right_index], dataY[right_index])))
           
    def query(self, points):
        # S={test points} given the model we built
        
        # Parameters:
        #     points: A numpy ndarray of test queries

        # Returns: 
        #     preds: A numpy 1D array of estimated values  

        predictions_query = []
        predictions_query = [self.prdctd_tree_search(point, row=0) for point in points]
        return np.asarray(predictions_query)

    def prdctd_tree_search(self, point, row):
        # Recursively searches the decision tree matrix and returns a predicted value for point

        # Parameters:
        #       point: Test-query -> 1D Array
        #       row: the decision tree matrix to search row
    
        # Returns 
        #       pred: Value -> Predicted. 

        # feature on 
        #       row & its corrosponding spliiting value. 
        
        ftur, split_val = self.tree[row, 0:2]
        
        # If splitting value of feature is -1, 
        #               --> return leaf. 
        if ftur == -1:
            return split_val

        # If the corresponding feature's value from point == split_val, 
        #                       --> left tree
        elif (point[int(ftur)] == split_val):
            pred = self.prdctd_tree_search(point, row + int(self.tree[row, 2]))
        
        # If the corresponding feature's value from point < split_val, 
        #                       --> left tree                
        elif (point[int(ftur)] < split_val):
            pred = self.prdctd_tree_search(point, row + int(self.tree[row, 2]))
        
        # Otherwise, go to the right tree
        else:
            pred = self.prdctd_tree_search(point, row + int(self.tree[row, 3]))
        
        return pred   
            
    def addEvidence(self, dataX, dataY):
        # feeding_data_to_learn_better

        # Params:
        #   dataX : A numpy ndarray -> x values at each node
        #   dataY : A numpy 1d array -> y values at each node

        # Returns: 
        #   Tree: A mumpy ndarray.
                
        if self.verbose:
            self.get_learner_info()

        #  If self.tree is not currently None -> append new_tree(=self.split_on_best_feature(dataX, dataY)) to self.tree
        # else:
        #       assign new_tree to it
        if self.tree is not None:
            self.tree = np.vstack((self.tree, self.split_on_best_feature(dataX, dataY)))
   
        else:            
            self.tree = self.split_on_best_feature(dataX, dataY) # Equivalent to new_tree

        # if row count == 1:
        #           expand tree to a numpy ndarray for consistency
        if len(self.tree.shape) - 1 == 0:
            self.tree = np.expand_dims(self.tree, axis=0)
    
    def get_learner_info(self):        
        if self.tree is None:
           print ("A moment of silence for the Tree and the Data")
        else:
             # Create a dataframe from tree for a user-friendly view
            df_tree = pd.DataFrame(self.tree, columns=["factor", "split_val", "left", "right"])
            df_tree.index.name = "node"
            print (df_tree)
        print ("")

if __name__=="__main__":
    # Some data to test the DTLearner
    x0 = np.array([0.872, 0.730, 0.575, 0.729, 0.620, 0.265, 0.500, 0.320])
    x1 = np.array([0.325, 0.385, 0.495, 0.569, 0.647, 0.647, 0.689, 0.750])
    x2 = np.array([9.150, 10.890, 9.390, 9.799, 8.392, 11.777, 10.444, 10.001])
    
    x = np.array([x0, x1, x2]).T
    
    y = np.array([4.000, 5.000, 6.000, 5.000, 3.000, 8.000, 7.000, 6.000])

    # Create a tree learner from given train X and y
    dtl = DTLearner(verbose=True, leaf_size=1)
    dtl.addEvidence(x, y)

    dtl2 = DTLearner(tree=dtl.tree)

    # dtl2 should have the same tree as dtl
    assert np.any(dtl.tree == dtl2.tree)

    dtl2.get_learner_info()

    # Modify the dtl2.tree and assert that this doesn't affect dtl.tree
    dtl2.tree[0] = np.arange(dtl2.tree.shape[1])
    assert np.any(dtl.tree != dtl2.tree)

    # Query with dummy data
    dtl.query(np.array([[1, 2, 3], [0.2, 12, 12]]))

    # Another Dataset

    x2_2 = np.array([
     [  0.26,    0.63,   11.8  ],
     [  0.26,    0.63,   11.8  ],
     [  0.32,    0.78,   10.   ],
     [  0.32,    0.78,   10.   ],
     [  0.32,    0.78,   10.   ],
     [  0.735,   0.57,    9.8  ],
     [  0.26,    0.63,   11.8  ],
     [  0.61,    0.63,    8.4  ]])
        
    y2 = np.array([ 8.,  8.,  6.,  6.,  6.,  5.,  8.,  3.])
        
    dtl = DTLearner(verbose=True)
    dtl.addEvidence(x2_2, y2)
