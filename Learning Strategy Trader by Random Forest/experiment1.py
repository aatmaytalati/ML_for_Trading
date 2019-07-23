# Aatmay S. Talati
# atalati3

import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import indicators as ind

import StrategyLearner as sl
import marketsimcode as mktsim

def author(self):
    return 'atalati3'

def test():

    ind.plotgraph('B','Best Indicator Option',policy = ind.testPolicy,sd=date_for(2008,1,1), ed=date_for(2009,12,31))
    ind.plotgraph('C','Apply',sd=date_for(2008,1,1), ed=date_for(2009,12,31))
    slearner = sl.StrategyLearner()

    slearner.addEvidence(symbol='JPM',sd=date_for(2008,1,1), ed=date_for(2009,12,31),sv = 10000)
    ind.plotgraph('IS ST','RT',policy = slearner.testPolicy, sd=date_for(2008,1,1), ed=date_for(2009,12,31))

if __name__ == '__main__':
    test()

def date_for(var1, var2, var3):
    return dt.datetime(2008,1,1)
