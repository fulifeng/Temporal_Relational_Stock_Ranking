import argparse
from datetime import datetime
import json
import numpy as np
import operator
import os
import pandas as pd

class EOD_Preprocessor:
    def __init__(self, data_path, market_name):
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S'
        self.market_name = market_name

    def _read_EOD_data(self):
        self.data_EOD = []
        for index, ticker in enumerate(self.tickers):
            single_EOD = np.genfromtxt(
                os.path.join(self.data_path, self.market_name + '_' + ticker +
                             '_30Y.csv'), dtype=str, delimiter=',',
                skip_header=True
            )
            self.data_EOD.append(single_EOD)
            # if index > 99:
            #     break
        print('#stocks\' EOD data readin:', len(self.data_EOD))
        assert len(self.tickers) == len(self.data_EOD), 'length of tickers ' \
                                                        'and stocks not match'

    def _read_tickers(self, ticker_fname):
        self.tickers = np.genfromtxt(ticker_fname, dtype=str, delimiter='\t',
                                     skip_header=True)[:, 0]

    def _transfer_EOD_str(self, selected_EOD_str, tra_date_index):
        selected_EOD = np.zeros(selected_EOD_str.shape, dtype=float)
        for row, daily_EOD in enumerate(selected_EOD_str):
            date_str = daily_EOD[0].replace('-05:00', '')
            date_str = date_str.replace('-04:00', '')
            selected_EOD[row][0] = tra_date_index[date_str]
            for col in range(1, selected_EOD_str.shape[1]):
                selected_EOD[row][col] = float(daily_EOD[col])
        return selected_EOD

    '''
        Transform the original EOD data collected from Google Finance to a
        friendly format to fit machine learning model via the following steps:
            Calculate moving average (5-days, 10-days, 20-days, 30-days),
            ignoring suspension days (market open, only suspend this stock)
            Normalize features by (feature - min) / (max - min)
    '''
    def generate_feature(self, selected_tickers_fname, begin_date, opath,
                         return_days=1, pad_begin=29):
        trading_dates = np.genfromtxt(
            os.path.join(self.data_path, '..',
                         self.market_name + '_aver_line_dates.csv'),
            dtype=str, delimiter=',', skip_header=False
        )
        print('#trading dates:', len(trading_dates))
        # begin_date = datetime.strptime(trading_dates[29], self.date_format)
        print('begin date:', begin_date)
        # transform the trading dates into a dictionary with index
        index_tra_dates = {}
        tra_dates_index = {}
        for index, date in enumerate(trading_dates):
            tra_dates_index[date] = index
            index_tra_dates[index] = date
        self.tickers = np.genfromtxt(
            os.path.join(self.data_path, '..', selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        print('#tickers selected:', len(self.tickers))
        self._read_EOD_data()
        for stock_index, single_EOD in enumerate(self.data_EOD):
            # select data within the begin_date
            begin_date_row = -1
            for date_index, daily_EOD in enumerate(single_EOD):
                date_str = daily_EOD[0].replace('-05:00', '')
                date_str = date_str.replace('-04:00', '')
                cur_date = datetime.strptime(date_str, self.date_format)
                if cur_date > begin_date:
                    begin_date_row = date_index
                    break
            selected_EOD_str = single_EOD[begin_date_row:]
            selected_EOD = self._transfer_EOD_str(selected_EOD_str,
                                                  tra_dates_index)

            # calculate moving average features
            begin_date_row = -1
            for row in selected_EOD[:, 0]:
                row = int(row)
                if row >= pad_begin:   # offset for the first 30-days average
                    begin_date_row = row
                    break
            mov_aver_features = np.zeros(
                [selected_EOD.shape[0], 4], dtype=float
            )   # 4 columns refers to 5-, 10-, 20-, 30-days average
            for row in range(begin_date_row, selected_EOD.shape[0]):
                date_index = selected_EOD[row][0]
                aver_5 = 0.0
                aver_10 = 0.0
                aver_20 = 0.0
                aver_30 = 0.0
                count_5 = 0
                count_10 = 0
                count_20 = 0
                count_30 = 0
                for offset in range(30):
                    date_gap = date_index - selected_EOD[row - offset][0]
                    if date_gap < 5:
                        count_5 += 1
                        aver_5 += selected_EOD[row - offset][4]
                    if date_gap < 10:
                        count_10 += 1
                        aver_10 += selected_EOD[row - offset][4]
                    if date_gap < 20:
                        count_20 += 1
                        aver_20 += selected_EOD[row - offset][4]
                    if date_gap < 30:
                        count_30 += 1
                        aver_30 += selected_EOD[row - offset][4]
                mov_aver_features[row][0] = aver_5 / count_5
                mov_aver_features[row][1] = aver_10 / count_10
                mov_aver_features[row][2] = aver_20 / count_20
                mov_aver_features[row][3] = aver_30 / count_30

            '''
                normalize features by feature / max, the max price is the
                max of close prices, I give up to subtract min for easier
                return ratio calculation.
            '''
            pri_min = np.min(selected_EOD[begin_date_row:, 4])
            price_max = np.max(selected_EOD[begin_date_row:, 4])
            print(self.tickers[stock_index], 'minimum:', pri_min,
                  'maximum:', price_max, 'ratio:', price_max / pri_min)
            if price_max / pri_min > 10:
                print('!!!!!!!!!')
            # open_high_low = (selected_EOD[:, 1:4] - price_min) / \
            #                 (price_max - price_min)
            mov_aver_features = mov_aver_features / price_max

            '''
                generate feature and ground truth in the following format:
                date_index, 5-day, 10-day, 20-day, 30-day, close price
                two ways to pad missing dates:
                for dates without record, pad a row [date_index, -1234 * 5]
            '''
            features = np.ones([len(trading_dates) - pad_begin, 6],
                               dtype=float) * -1234
            # data missed at the beginning
            for row in range(len(trading_dates) - pad_begin):
                features[row][0] = row
            for row in range(begin_date_row, selected_EOD.shape[0]):
                cur_index = int(selected_EOD[row][0])
                features[cur_index - pad_begin][1:5] = mov_aver_features[
                    row]
                if cur_index - int(selected_EOD[row - return_days][0]) == \
                        return_days:
                    features[cur_index - pad_begin][-1] = \
                        selected_EOD[row][4] / price_max

            # # write out
            # np.savetxt(os.path.join(opath, self.market_name + '_' +
            #                         self.tickers[stock_index] + '_' +
            #                         str(return_days) + '.csv'), features,
            #            fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    desc = "pre-process EOD data market by market, including listing all " \
           "trading days, all satisfied stocks (5 years & high price), " \
           "normalizing and compansating data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-path', help='path of EOD data')
    parser.add_argument('-market', help='market name')
    args = parser.parse_args()

    if args.path is None:
        args.path = '../data/google_finance'
    if args.market is None:
        args.market = 'NASDAQ'

    processor = EOD_Preprocessor(args.path, args.market)
    processor.generate_feature(
        processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv',
        datetime.strptime('2012-11-19 00:00:00', processor.date_format),
        os.path.join(processor.data_path, '..', '2013-01-01'), return_days=1,
        pad_begin=29
    )

    processor.generate_feature(
        processor.market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv',
        datetime.strptime('2012-11-19 00:00:00', processor.date_format),
        os.path.join(processor.data_path, '..', '2013-01-01'), return_days=1,
        pad_begin=29
    )