import numpy as np
import csv
import os
import time
import datetime
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
from fastdtw import fastdtw
from joblib import Parallel, delayed
from numpy.lib import stride_tricks
import pickle
import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--index', action='store', dest='index', default = 'SP500')
parser.add_argument('--backtest', action='store', dest='backtest')
parser.add_argument('--base_dir', action='store', dest='base_dir', default = '/hdd/quant/')
parser.add_argument('--bitcoin_file', action='store', dest='bitcoin_file', default = 'bitcoin.csv')
'''
index options:
SP500 (S&P500, large capitalization)
SP400 (mid capitalization)
SP600 (small capitalization)
'''
args = parser.parse_args()

results = []

#load .csv file containing daily bitcoin prices 
bitcoin_prices = []
first_line = True
with open(args.base_dir + args.bitcoin_file) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        if first_line:
            first_line = False
            continue
        bitcoin_prices.append(row[1])

#subsample weekly prices from daily prices to speed up computations
bitcoin_prices = np.array(map(float,bitcoin_prices[0:-1:7]))

if args.backtest != None:
    bitcoin_prices = bitcoin_prices[0:-int(args.backtest)]

#plt.plot(bitcoin_prices)
#plt.show()

#load .csv file containing daily stock prices
data_dir = args.base_dir + args.index + '/'
file_list = os.listdir(data_dir)

count = 0
for f in file_list:
    ticker = f.split('.csv')[0]
    print "ticker: {} ({}/{})".format(ticker, count, len(file_list))
    stock_prices = []
    first_line = True
    with open(data_dir + f) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if first_line:
                first_line = False
                continue
            if row[6] not in ['0', '-0', '-NaN', 'NaN', '-Inf', 'Inf']:
                if args.index != 'FC':
                    stock_prices.append(row[6])
                else:
                    #no need for split and dividend adjustments for futures and commodities
                    stock_prices.append(row[4])

    #convert daily to weekly stock data
    stock_prices = np.array(map(float,stock_prices[0:-1:7]))
    if args.backtest != None:
        stock_prices = stock_prices[0:-int(args.backtest)]
    best_distance = 99999999
    best_template = (0,0)
    stride = 2
    #use only companies that have existed for a longer period of time than has bitcoin
    if len(stock_prices) > len(bitcoin_prices):
        prev = time.time()
        #brute force pattern matching
        ticker_results = []
        for template_window in range(100, min(501,len(stock_prices)), 200):
            print template_window
            batch_results = []
            for j in range((len(stock_prices)-template_window)/stride):
                #templates are sliding windows taken from the stock prices array
                template = stock_prices[j*stride:j*stride+template_window]
                #compute z-normalization for scale-invariant pattern matching
                shapelet = zscore(bitcoin_prices)
                template = zscore(template)
                #compute euclidean distances with Dynamic Time Warping
                distance, path = fastdtw(template, shapelet, dist=euclidean)
                #scale distances by a factor proportional to the template window size
                distance *= 1./(np.sqrt(template_window/100.))
                #save the best-so-far information
                if distance < best_distance:
                    best_distance = distance
                    best_template = (j, template_window)
        results.append((best_distance, best_template, ticker))
        curr = time.time()
        print str(datetime.timedelta(seconds=curr-prev))
    count += 1

#sort dictionary based on best_distance values
sorted_results = sorted(results)
print len(sorted_results)
#save results to disk
if args.backtest != None:
    fname = args.base_dir+args.index+'_backtest.pickle'
else:
    fname = args.base_dir+args.index+'.pickle'
with open(fname, 'w') as f:
    pickle.dump(sorted_results, f)

