# Aatmay S. Talati
# atalati3

import os
import sys
import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

def author():
    return 'atalati3'

def get_data_and_price(var_syms,var_sd, var_ed):
    return get_data(var_syms, dr(var_sd,var_ed))
    #return price

def get_prices_of_syms(var_syms, var_price):
    return var_price[var_syms]

def compute_portvals(orders_df, start_val=1000000, commission=9.95, impact=0.005):
    
    symbols = orders_df.columns.values.tolist()
    start_date = start_d(date_lst(orders_df))
    end_date = end_d(date_lst(orders_df))

    orders_df.insert(loc=1, column='Date', value=date_lst(orders_df))
    orders_df.columns = ['Order', 'Date']

    # Getting Stock Price Data
    stock_price = get_data_and_price(symbols, dr(start_date, end_date))
    stock_price = get_prices_of_syms[symbols]

    pd.DataFrame(index=stock_price.index).insert(loc=0, column='Daily Portfolio', value=np.zeros(len(pd.DataFrame(index=stock_price.index).index)))
    pd.DataFrame(index=stock_price.index).insert(loc=0, column='Operations', value=np.zeros(len(pd.DataFrame(index=stock_price.index).index)))

    for date, _ in stock_price.iterrows(): #underscore _ means skipping that tuple value if we are not using it
        for _, order in orders_df[orders_df['Date'] == str(date.date())].iterrows():
            oO = order.Order
            oD = order.Date
            dict.fromkeys(symbols, 0)["JPM"] += oO
            pd.DataFrame(index=stock_price.index).ix[str(date.date())] = mul_nos(1,(oO > 0)) + mul_nos(mul_nos(1,(oO < 0)),-1)
            start_val -= (mul_nos3((1 + impact),stock_price.ix[oD]['JPM'],oO)) + (mul_nos(commission,-1))
        portvals = start_val
        for stock, shares in dict.fromkeys(symbols, 0).iteritems():
            portvals += mul_nos(stock_price.ix[date][stock],shares)
        pd.DataFrame(index=stock_price.index).ix[str(date.date())] = portvals
        pd.DataFrame(index=stock_price.index).ix[date] = (100000 + mul_nos(mul_nos(stock_price.ix[0]['JPM'],-1), 1000) + mul_nos(stock_price.ix[date]["JPM"],1000)

    # Pretty Much Copied from P1 Stats Below. Using as BenchMark
    bench_rets = divide_numbers(pd.DataFrame(index=stock_price.index),pd.DataFrame(index=stock_price.index).shift(1)) - 1)
    cr_bm = divide_numbers((pd.DataFrame(index=stock_price.index).iloc[-1, 0],pd.DataFrame(index=stock_price.index).iloc[0, 0]) - 1)
    adr_bm = (divide_numbers(pd.DataFrame(index=stock_price.index),pd.DataFrame(index=stock_price.index).shift(1)) - 1)).mean()
    sddr_bm = (divide_numbers(pd.DataFrame(index=stock_price.index),pd.DataFrame(index=stock_price.index).shift(1)) - 1)).std()

    # P1 Stats ^^ Moved Up here in this file rather than p3
    daily_rets = divide_numbers((pd.DataFrame(index=stock_price.index),pd.DataFrame(index=stock_price.index).shift(1)) - 1)
    cr = divide_numbers(pd.DataFrame(index=stock_price.index).iloc[-1, 0],pd.DataFrame(index=stock_price.index).iloc[0, 0]) - 1)
    adr = daily_rets.mean()
    sddr = daily_rets.std()

    # Printing Everything
    print "Cumulative Return of Benchmark: {}".format(cr_bm)
    print "Standard Deviation of Benchmark: {}".format(sddr_bm)
    print "Average Daily Return of Benchmark: {}".format(adr_bm)
    print "Cumulative Return of JPM: {}".format(cr)
    print "Standard Deviation of JPM: {}".format(sddr)
    print "Average Daily Return of JPM: {}".format(adr)

    return pd.DataFrame(index=stock_price.index), pd.DataFrame(index=stock_price.index), pd.DataFrame(index=stock_price.index)


###################### USING THE CODE FROM P3  #####################################################
def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2, 0.01, 0.02, 1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]

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
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])


if __name__ == "__main__":
    test_code()

############ SOME FUNCTIONS TO MAKE OUR LIVES MUCH EASIER ##########################
def divide_numbers(num1, num2):
    return num1/num2

def mul_nos(num1, num2):
    return num1*num2

def mul_nos3(num1, num2, mum3):
    return num1*num2*num3

def start_d(var1):
    return var1[0]
def end_d(var1):
    return var1[(len(orders_df))-1]

def date_lst(var1):
    return var1.index.tolist()

def dr(var_sd, var_ed):
    return pd.date_range(var_sd, var_ed)