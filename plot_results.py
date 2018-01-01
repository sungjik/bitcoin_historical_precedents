import numpy as np
import os
import csv
import pickle
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
from fastdtw import fastdtw
import argparse
import datetime
import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt
import matplotlib.dates as dates

parser = argparse.ArgumentParser()
parser.add_argument('--index', action='store', dest='index', default = 'SP500')
parser.add_argument('--num_best', action='store', dest='num_best', default = '5')
parser.add_argument('--backtest', action='store', dest='backtest')
parser.add_argument('--base_dir', action='store', dest='base_dir', default = '/hdd/quant/')
parser.add_argument('--bitcoin_file', action='store', dest='bitcoin_file', default = 'bitcoin.csv')

'''
index options:
SP500 (S&P500, large capitalization)
SP400 (mid capitalization)
SP600 (small capitalization)
Composite
'''
args = parser.parse_args()
num_best = int(args.num_best)
count = 2
tail_size = 300
fig = plt.figure(facecolor="white", figsize=(10.0, float(num_best)*2.5))

if args.backtest != None:
    tail_size = int(args.backtest)
    fig = plt.figure(facecolor="white")

stride = 2

#load .csv file containing daily bitcoin prices 
bitcoin_prices = []
bitcoin_dates = []
first_line = True
with open(args.base_dir+args.bitcoin_file) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        if first_line:
            first_line = False
            continue
        bitcoin_prices.append(row[1])
        bitcoin_dates.append(row[0].split(' ')[0])

#convert daily to weekly bitcoin data
bitcoin_prices = np.array(map(float,bitcoin_prices[0:-1:7]))
bitcoin_dates = bitcoin_dates[0:-1:7]

if args.backtest != None:
    bitcoin_prices_test = bitcoin_prices[0:-int(args.backtest)]
    bitcoin_dates_test = bitcoin_dates[0:-int(args.backtest)]

sorted_results = []
if args.index == 'Composite':
    for index in ['SP500', 'SP400', 'SP600']:
        if args.backtest != None:
            fname = args.base_dir+index+'_backtest.pickle'
        else:
            fname = args.base_dir+index+'.pickle'
        with open(fname, 'r') as f:
            index_results = pickle.load(f)
        for i in range(len(index_results)):
            index_results[i] = tuple(list(index_results[i]) + [index])
        sorted_results += index_results
    sorted_results = sorted(sorted_results)
else:
    if args.backtest != None:
        fname = args.base_dir+args.index+'_backtest.pickle'
    else:
        fname = args.base_dir+args.index+'.pickle'
    with open(fname, 'r') as f:
        sorted_results = pickle.load(f)

    for i in range(len(sorted_results)):
        sorted_results[i] = tuple(list(sorted_results[i]) + [args.index])
            
best = sorted_results[0:num_best]
predicted_prices = np.array([0.0]*tail_size)
for stock in best:
    index = stock[3]
    ticker = stock[2]
    offset = stock[1][0]
    window = stock[1][1]
    stock_prices = []
    stock_dates = []
    first_line = True
    with open(args.base_dir+index+'/'+ticker+'.csv', 'r') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if first_line:
                first_line = False
                continue
            if row[6] not in ['0', 'NaN', '-Inf', 'Inf']:
                if args.index != 'FC':
                    stock_prices.append(row[6])
                else:
                    #no need for split and dividend adjustments for futures and commodities
                    stock_prices.append(row[4])
                stock_dates.append(row[0])
                    
    stock_prices = np.array(map(float,stock_prices[0:-1:7]))
    stock_dates = stock_dates[0:-1:7]
    tail = np.array(stock_prices[offset*stride+window+1:offset*stride+window+tail_size+1])
    #tail = zscore(tail)
    if len(tail) >= tail_size:
        predicted_prices += tail
    
    if args.backtest == None:
        ax = fig.add_subplot(num_best+1,1,count)
        ax.plot_date(dates.datestr2num(stock_dates[:offset*stride+1]), stock_prices[:offset*stride+1], fmt="k", xdate=True, label=ticker+'('+index+')'+' historical')
        ax.plot_date(dates.datestr2num(stock_dates[offset*stride:offset*stride+window+1]), stock_prices[offset*stride:offset*stride+window+1], fmt="r", xdate=True, label='similar to BTC')
        ax.plot_date(dates.datestr2num(stock_dates[offset*stride+window:]), stock_prices[offset*stride+window:], fmt="k", xdate=True)
        plt.legend(loc=0)
    count += 1
    
#get averages of the tails and compute predicted prices
predicted_prices /= num_best
predicted_prices = zscore(predicted_prices)
#align beginning price of prediction with end price of bitcoin
diff = predicted_prices[0] - np.array(zscore(bitcoin_prices))[-1]
if diff < 0:
    predicted_prices -= diff
else:
    predicted_prices += diff
    

#compute inverse z-normalization on predictions using std and mean of bitcoin prices
predicted_prices = np.std(bitcoin_prices)*predicted_prices+np.mean(bitcoin_prices)

if args.backtest != None:
    predicted_dates = dates.datestr2num(bitcoin_dates[-tail_size:])
    ax1 = fig.add_subplot(1,1,1)
    distance, path = fastdtw(bitcoin_prices[-tail_size:], predicted_prices, dist=euclidean)

else:
    ax1 = fig.add_subplot(num_best+1,1,1)
    predicted_dates = np.arange(tail_size)*7+dates.datestr2num(bitcoin_dates[-1])

ax1.plot_date(dates.datestr2num(bitcoin_dates), bitcoin_prices, fmt="r", xdate=True, label='BTC historical')
ax1.plot_date(predicted_dates, predicted_prices, fmt="b", xdate=True, label='BTC predicted')
plt.legend(loc=0)

#plt.show()
fname = args.base_dir+args.index+'_'+str(num_best)+'.png'
if args.backtest != None:
    fname = args.base_dir+args.index+'_'+str(num_best)+'_backtest.png'
    
plt.savefig(fname, dpi=150, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
