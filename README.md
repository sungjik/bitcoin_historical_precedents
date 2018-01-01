# bitcoin_historical_precedents
https://sungjik.wordpress.com/2018/01/01/searching-for-historical-precedents-of-bitcoin-by-pattern-matching/

1. To download S&P 1500 data, run downloadData.m inside the folder googleFinanceCrawler
2. To mine for precedents, run
$ python find_precedents.py --index SP500 --base_dir bitcoin_historical_precedents/
3. To plot results, run
$ python plot_results.py --index Composite --base_dir bitcoin_historical_precedents/
4. Get latest bitcoin prices from https://www.coindesk.com/price/
