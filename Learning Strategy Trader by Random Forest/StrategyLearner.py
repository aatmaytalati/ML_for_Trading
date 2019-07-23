"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""
# Aatmay S. Talati
# atalati3

import os
import sys
import datetime as dt
import numpy as np
import pandas as pd
import util as ut
import RTLearner as rt
import indicators
import itertools
import BagLearner as bagL
 
def cumulativeSum(input):
    print ("Sum :", list(itertools(input)))

def author(self):
    return 'atalati3' #GT Username

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.learner = rt.RTLearner(leaf_size=5)
        self.lb = 14
   
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000, \
        impact = 0):

        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms=[symbol]
        dates = dr(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices
        # example use with new colname

        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume

############################# AS IT IS ABOVE THIS LINE #######################################################

        self.indicators = np.cumsum(indicators.testPolicy(symbol,sd,ed,sv,impact = 0))
        temp1 = pd.concat([self.dataset_addind(sd,ed,symbol,self.lb),self.indicators],axis = 1).ix[:,0:-1]
        temp2 = pd.concat([self.dataset_addind(sd,ed,symbol,self.lb),self.indicators],axis = 1).ix[:,-1]
        self.learner.addEvidence(temp1, temp2)


# Importing indicators from indicatators.py
    def dataset_addind(self,sd,ed,symbol,last_inst): # Callling indicators from indicatators.py
        sma = indicators.sma(sd , ed ,syms = [symbol],last_inst = self.lb, ratio = False)
        bollinger_bands = indicators.bollinger_bands(sd , ed ,syms = [symbol],last_inst = self.lb)
        relative_strength = indicators.relative_strength(sd , ed ,syms = [symbol],last_inst = self.lb)
        sochastic_oscillator = indicators.sochastic_oscillator(sd , ed ,syms = [symbol],last_inst = self.lb)
        df = pd.concat([sma,bollinger_bands[[symbol]],sochastic_oscillator[[symbol]]],axis = 1)
        return df

###################### TESTING THE POLICY >> USING THE CODE GIVEN ####################################
    
    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000,impact = None):
        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = dr(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols

######################################################################################        
    
        trades = copy_var(self.learner.query(self.dataset_addind(sd,ed,symbol,self.lb)))
        trades.values[1:] = self.learner.query(self.dataset_addind(sd,ed,symbol,self.lb)).values[1:] + mul_nos(self.learner.query(self.dataset_addind(sd,ed,symbol,self.lb)).values[:-1], -1)
        trades.values[0] = self.learner.query(self.dataset_addind(sd,ed,symbol,self.lb)).values[0]
        trades.columns = [symbol]
 #################################################################################       

        if self.verbose: print type(trades)
        if self.verbose: print trades
        if self.verbose: print prices_all
        print trades
        return trades

if __name__=="__main__":
    print ("Have a great semester!")
    addEvidence(symbol="ML4T-220",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    testPolicy(symbol="ML4T-220",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
    print "GREAT MINDS THINK ALIKE"

    addEvidence(symbol="AAPL",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    testPolicy(symbol="AAPL",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
    print "I KNOW GREAT MINDS THINK ALIKE"

    addEvidence(symbol="UNH",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    testPolicy(symbol="UNH",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
    print "I'M 100% POSTIVE THAT GREAT MINDS THINK ALIKE"
    
    addEvidence(symbol="SINE_FAST_NOISE",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000)
    testPolicy(symbol="SINE_FAST_NOISE",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000)
    print "NOW I'VE AN ASSURANCE THAT GREAT MINDS THINK ALIKE"
    
########################## FEW FUNCTIONS BELOW TO MAKE OUR LIVES EASIER ################################
def copy_var(var):      # Same as Indicator File
    return var.copy()

def hldng(self,var1): # Same as Indicator File
    abc = self.learner.query(var1)
    return abc

def mul_nos(num1, num2): # Same as Indicator File
    return num1*num2

def dr(var_sd, var_ed):
    return pd.date_range(var_sd, var_ed)

