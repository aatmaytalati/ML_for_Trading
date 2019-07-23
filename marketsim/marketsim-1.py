"""MC2-P1: Market simulator."""

import sys
import os
import pandas as pd
import numpy as np
import math
import types
from util import get_data

total_trading_days_in_a_year = 252

def author():
    return 'atalati3'

def sd(var1):
    return var1.index[0]

def ed(var1):
    return var1.index[-1]

def Date_Range(var1):
    return pd.date_range(sd(var1), ed(var1))

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    
    # Opening the file. 
    f = None    
    while (f == None):
        if type(orders_file) == types.FileType:
            f = orders_file
        elif type(orders_file) == types.StringType:
            f = open(orders_file, 'r') # Opening file as a text file.
        else:
            print "Error 404"

    order_df = (pd.read_csv(f, index_col='Date', parse_dates=True)) # Reading CSv file like p1. 
    print (order_df).sort_index(inplace=True) # Check point
    symbol_names = list(set(order_df['Symbol'])) # Getting a list of symbol Names. 
    # Referance : https://stackoverflow.com/questions/6828722/python-set-to-list
  
    prices = get_data(symbol_names, Date_Range(order_df))
    portvals = pd.DataFrame(data=np.arange(len(prices)), index=prices.index, columns=['val'])
    print portvals

    s_lst = {}
    for name in symbol_names:
        s_lst[name] = 0
        
    order_idx = 0
    for i in range(len(portvals)):
        x_temp = len(order_df.index)
        i_temp = portvals.index[i]
        
        while order_idx < x_temp and i_temp == order_df.index[order_idx]:
            sign_tmp = 1 if order_df['Order'].iloc[order_idx] == 'BUY' else -1
            
            temp = sign_tmp * order_df['Shares'].iloc[order_idx]
            temp_s = s_lst[order_df['Symbol'].iloc[order_idx]]
            s_lst[order_df['Symbol'].iloc[order_idx]] += temp
            
            temp_si = sign_tmp 
            temp_si = temp_si + impact
            temp_i2 = prices[order_df['Symbol'].iloc[order_idx]].iloc[i]

            start_val = (start_val - (temp_si * order_df['Shares'].iloc[order_idx] * temp_i2))-commission
            order_idx = order_idx +1
            
        portvals.iloc[i] = start_val
        for k in s_lst:
            temp_x = s_lst[k]
            temp_y = prices[k].iloc[i]
            temp_xy = (temp_x * temp_y) 
            portvals.iloc[i] += temp_xy
    print portvals[:20]
    return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/order_short.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # States form P1

    dates = Date_Range(portvals)   
    
    prices_all = get_data(['GOOG'], dates)
    
    cum_ret = (portvals[-1] / portvals[0])
    cum_ret = cum_ret - 1
    
    daily_returns =  ((portvals / portvals.shift(1)) - 1)[1:]
  
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    
    sharpe_ratio = math.sqrt(total_trading_days_in_a_year) * (avg_daily_ret)
    sharpe_ratio = sharpe_ratio / std_daily_ret


    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
