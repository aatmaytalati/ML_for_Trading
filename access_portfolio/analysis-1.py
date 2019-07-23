"""Analyze a portfolio.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""
import os
import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import math
import matplotlib.pyplot as plt

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Step 1: Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Step 2: Normalization of Prices
    normed_frame = prices / prices.ix[0,:]  # Normalize by first row. 
    # print normed_frame  # Checker 

    # Get daily portfolio value
    port_val = prices_SPY # add code here to compute daily portfolio values

    # Udacity Vidoes: ML4T- CS 7646 by Tucker Balch Lesson 8 Video 2
    # Step 3: allocated = Multiply by allocation (a vector)
    # Step 4: pos_vals = allocated * start_Val
    # Step 5: Daily Portfolio values = pos_vals.sum(axis = 1) => summing all values of axis number 1
    # Step 6: Daily Return

    alloctd = normed_frame * allocs # Step 3

    start_Val = sv                           # \
    pos_val = start_Val*alloctd     		 # / Step 4 : Position Value

    daily_port_value = pos_val.sum(axis=1) # step 5

    # Step 6:
    # Referance: https://classroom.udacity.com/courses/ud501/lessons/4242038556/concepts/41998985400923
    copy_of_daily_port_value = daily_port_value.copy()  
    daily_return = copy_of_daily_port_value  # made a copy oof daily portfolio value
    daily_return[1:] = (daily_return[1:] / daily_return[:-1].values)-1
    daily_return.ix[0] = 0  # First day there are no change. so it always stays 0.

    # Get portfolio statistics (note: std_daily_ret = volatility)
    # Resource for this chunk of function: https://classroom.udacity.com/courses/ud501/lessons/4242038556/concepts/41998985400923
    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats


    # def get_formulas (var_cr, var_adr, var_sddr, var_sr):
    # Cumulative Return (cr)
    cr = (daily_port_value[-1]/daily_port_value[0])-1

    #Average Daily Return (adr)
    adr = daily_return[1:].mean()

    # Standard Daviation of Daily Return (sddr)
    sddr = daily_return[1:].std() # Volatality

    # Sharp Ratio (sr)
    sr = math.sqrt(sf) * (daily_return[1:] - rfr).mean() / sddr

    #get_formulas(cr, adr, sddr, sr)

    # def plot_it(var):
    #     var.plot() #plotting the given variable
    #     plt.show() # telling compiler to disaply the plot

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here 
        # referance: https://github.com/jrajamaki/ML4T/blob/janne/mc1_p1/analysis.py
      
        normed_spy = prices_SPY / prices_SPY.ix[0,:]
        df_temp = pd.concat([alloctd.sum(axis=1), normed_spy], keys=['Portfolio', 'SPY'], axis=1)
        df_temp.plot()
        plt.draw()
        # pass
    # Add code here to properly compute end value
    # ev = sv
    ev = ((sv * cr) + sv)
    return cr, adr, sddr, sr, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = False)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
