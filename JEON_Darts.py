import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.utils.utils import SeasonalityMode
from darts.models import ExponentialSmoothing
from darts.models import AutoARIMA
from darts.models import NaiveSeasonal
from DataPreprocess import DataPreprocess
import matplotlib.pyplot as plt


def read_sy_df():
    # 데이터프레임 전처리용 클래스
    dp = DataPreprocess()
    # 전처리된 데이터프레임
    df = dp.df_prcd

    return df


def original_df():
    filepath = './dataset/'
    df = pd.read_csv(filepath + 'dataset.csv')

    return df


def NG_OK(_data):
    if _data == 'OK':
        return 1
    else:
        return 0


def Example_sub():
    df = read_sy_df()
    series = TimeSeries.from_dataframe(df, 'DATE_TIME', 'MELT_WEIGHT')
    train, val = series.split_before(pd.Timestamp(2020, 4, 20, 00, 00, 00))

    # model = ExponentialSmoothing()
    # model = ExponentialSmoothing(seasonal=SeasonalityMode.NONE)
    model = AutoARIMA()

    model.fit(train)
    prediction = model.predict(len(val))

    series.plot()
    prediction.plot(label='forecast', lw=3)

    # series.plot(label='actual')
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)

    plt.legend()
    plt.show()


def Darts_Example_usage():
    df = pd.read_csv('./AirPassengers.csv')
    series = TimeSeries.from_dataframe(df, 'Month', '#Passengers')
    train, val = series.split_before(pd.Timestamp('19580101'))
    model = ExponentialSmoothing()
    model.fit(train)
    prediction = model.predict(len(val))
    series.plot(label='actual')
    prediction.plot(label='forecast', lw=3)
    plt.legend()
    plt.show()


def sub_test():
    dataframe = read_sy_df()
    # print(dataframe["DATE_TIME"])
    # dataframe['TAG'] = dataframe['TAG'].apply(NG_OK)

    df = dataframe[['DATE_TIME', 'MELT_WEIGHT']]
    series = TimeSeries.from_dataframe(df, 'DATE_TIME', 'MELT_WEIGHT')
    print(series)
    # df.loc[df['MELT_WEIGHT'] >= 1000, 'MELT_WEIGHT'] = 0
    #
    # df.replace(0, np.nan, inplace=True)
    # df = df.fillna(method="bfill")
    # plt.figure(figsize=(12, 5))
    # df.plot()
    # plt.show()

    # series = TimeSeries.from_dataframe(dataframe, 'DATE_TIME', 'MELT_WEIGHT')
    #

    #
    # train, val = series.split_before(pd.Timestamp(2020, 4, 20, 00, 00, 00))
    #
    # # model = ExponentialSmoothing()
    # # model = NaiveSeasonal(train)
    # model.fit(train)
    # prediction = model.predict(len(val))
    #
    # series.plot()
    # prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
    # plt.legend()


if __name__ == '__main__':
    # Darts_Example_usage()
    Example_sub()
