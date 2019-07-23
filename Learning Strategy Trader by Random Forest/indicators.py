# implements your indicators as functions that operate on dataframes. The "main" code in indicators.py should generate the charts that illustrate your indicators in the report.

# Aatmay S. Talati
# atalati3

import sys
import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import math

def author(self):
    return 'atalati3' #GT Username

##################### FEW FUNCTIONS WHICH MAKES OUR LIVES EASIER ################################
def dot_price(var1):
    var1.columns += '.price'
    return var1.columns

def dot_rm(var1):
    var1.columns += '.r_m'
    return var1.columns

def get_data_and_price(var_syms,var_sd, var_ed):
    return get_data(var_syms, dr(var_sd,var_ed))
    #return price

def get_prices_of_syms(var_syms, var_price):
    return var_price[var_syms]

def t_bnd(var1, var2):
    return var1+mul_nos(2,var2)

def d_r_v(var1, var2):
    var1.values[1:,:]= var2.values[1:,:]-var2.values[:-1,:]
    var1.values[0,:] = np.nan
    return var1

def d_r_v2(var1,var2,var3):
    var1.values[:-1,:] = var2.values[:-1,:] - var3.values[1:,:]
    var1.values[-1,:] = np.inf

def d_r_v3(var1, var2, var3 ):
    var1.values[1:,:] = var2.values[:-1,:] - var3.values[1:,:]
    var1.values[0,:] = np.inf

def trd_eq_h(var1, var2,var3):
    var1.values[1:,:] = var2.values[1:,:] + mul_nos(var2.values[:-1,:],-1)
    var1.values[0,:] = var2.values[0,:]
    var1.columns = [var3]
    return var1

######################################################################################
    
# Sample Moving Average
# Referance: https://www.investopedia.com/video/play/moving-average/
def sma(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), \
    syms = ['JPM'],last_inst = 10, ratio = False): 

    price = get_data_and_price(syms, sd, ed)
    price = get_prices_of_syms(syms, price)

    sma = price.rolling(window = last_inst,min_periods=last_inst).mean()
    try_temp = divide_numbers(price,sma)
    
    boo = False
    if (mul_nos(ratio,boo)): 
        return sma
    else:
        dot_price(price)
        temp_var = concat_df([try_temp, price, sma],1)
        return temp_var

##########################################################################################
# Bollinger Bands(r)
# Referance: https://www.investopedia.com/walkthrough/forex/intermediate/level4/bollinger-bands.aspx

def bollinger_bands(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), \
    syms = ['JPM'],last_inst = 10):
    
    price = get_data_and_price(syms, sd, ed)
    price = get_prices_of_syms(syms, price)

    bb_sma = sma(sd,ed,syms)
    rolling_std = price.rolling(window = last_inst,min_periods=last_inst).std()
    
    t_bnd = bb_sma+ mul_nos(rolling_std,2)
    b_bnd = bb_sma+mul_nos(mul_nos(2,rolling_std),-1)
    bbp = divide_numbers(price+mul_nos(b_bnd,-1),(t_bnd + mul_nos(b_bnd,-1)))
  
    dot_price(price)
    temp_var = concat_df([bbp,t_bnd,b_bnd],1)
    return temp_var
    
def try_r(var1, var2,var3):
    var1.ix[:,:] = 0
    var1.values[var3:,:] = var2.values[var3:,:] + mul_nos(var2.values[:-var3,:],-1)
    return var1

################################################################################################
# Relative Strength
# Referance: https://www.investopedia.com/terms/r/relativestrength.asp
def relative_strength(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), \
    syms = ['JPM'],last_inst = 10):
    price = get_data_and_price(syms, sd, ed)
    price = get_prices_of_syms(syms, price)
    x = copy_var(price)
    
    daily_returns = x
    upper_vakro = x
    niche_khot = x
    temp_var_x = daily_returns[daily_returns >= 0]
    temp_var_y = daily_returns[daily_returns < 0]
    
    uppar_return = temp_var_x.fillna(0).cumsum()
    niche_return = mul_nos(temp_var_y.fillna(0).cumsum(),-1)
  
    try_r(upper_vakro, uppar_return, last_inst)
    try_r(niche_khot, niche_return, last_inst)
    
    rs = divide_numbers(upper_vakro,niche_khot)
    relative_strength = 100 + mul_nos(divide_numbers(100,(1+rs)), -1)
    relative_strength.fillna(100,inplace = True)
    
    temp_var = concat_df([relative_strength,upper_vakro,niche_khot],1)
    return temp_var

########################################################################################################
#Stochastic_Oscillator
# Description & Psudocode link: https://www.investopedia.com/terms/s/stochasticoscillator.asp

def sochastic_oscillator(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), \
    syms = ['JPM'],last_inst = 14):
    
    price = get_data_and_price(syms, sd, ed)
    price = get_prices_of_syms(syms, price)

    rolling_max = price.rolling(window = last_inst,min_periods=last_inst).max()
    rolling_min = price.rolling(window = last_inst,min_periods=last_inst).min()
    temp_var2 = mul_nos(rolling_min, -1)
    dl_r = rolling_max+temp_var2
    
    sochastic_oscillator = mul_nos((divide_numbers((price - rolling_min),dl_r)),2)
    dot_rm(rolling_max)

    temp_var = concat_df([sochastic_oscillator,rolling_max,rolling_min],1)
    return temp_var

########################################################################################################################################
# Pretty Much Same as StrategyLearner. Except for a Loop
def testPolicy(symbol = "AAPL",sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000,commission = 0, impact = None) :
    symbol = [symbol]
    dates = pd.date_range(sd, ed)
    prices = get_data(symbol, dates)  # automatically adds SPY
    prices = get_prices_of_syms(symbol, prices)
    x=copy_var(prices)

    if (impact is not None ) and (impact != 0):
        dbp, dsp, drb, drs = x
        dbp =  mul_nos(dbp,1) + mul_nos(dbp,impact)
        dsp = mul_nos(dsp,(1+ mul_nos(impact,-1)))
        d_r_v2(drb, dbp, dsp)
        d_r_v3(drs, dsp, dpb)
        daily_returns =  drb + mul_nos(drs,-1)
    else:
        daily_returns=x
        d_r_v(daily_returns, prices)

    h = x
    h.values[:-1,0] =[1000 if x > 0 else -1000 for x in d_r_Try(daily_returns,symbol)]
    h.values[-1,:] = 0
    
    trd = copy_var(h)
    trd_eq_h(trd, h, symbol)
    return trd

#########################################################################################################################################
# Functions to make our lives easier 
def divide_numbers(num1, num2):
    return num1/num2

def mul_nos(num1, num2):
    return num1*num2

def dr(var_sd, var_ed):
    return pd.date_range(var_sd, var_ed)

def concat_df(var1, var2):
    return pd.concat(var1,axis=var2)

def d_r_Try(var1,var2):
    return var1[var2].values[1:,]

def copy_var(var):
    return var.copy()