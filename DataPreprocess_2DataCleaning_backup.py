# pip install pyod
# pip install kats
# pip install tods
import datetime

import numpy as np

from DataPreProcess_1DataLoad import DataLoad
from DataPreprocess import DataPreprocess

# from pyod.utils.example import visualize
# from pyod.utils.data import evaluate_print
# from pyod.models.knn import KNN

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import RepeatVector
# from tensorflow.keras.layers import TimeDistributed

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from prophet import Prophet


from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from ThymeBoost import ThymeBoost as tb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import Errors

"""
데이터 클리닝
- 대상: noise, artifact, precision, bias, outlier, missing values, inconsistent value, duplicate data
- 진단
    - missing values 존재하지 않음
    - duplicate data 존재하지 않음
    
"""

class DataCleaning(DataPreprocess):
    def __init__(self):
        super().__init__()

    def no_missingValues(self):
        """
        원본 데이터베이스에 null 값이 있는지 확인.
        :return: True (null이 데이터프레임 내 하나도 없을 경우) or False (else)
        """
        result = True if self.df_org.isnull().sum() ==0 else False
        return result

    def no_duplicates(self):
        """
        원본 데이터베이스에 중복된 데이터가 있는지 확인.
        :return: True (null이 데이터프레임 내 하나도 없을 경우) or False (else)
        """
        result = True if self.df_org.duplicated().sum() == 0 else False
        return result

    def narrow_dataframe(self, df, cols):
        return df[cols]

    def plot_before_after(self, data_before, data_after):
        f, ax = plt.subplots(1, 1, figsize=(18, 9))
        ax.plot(data_before.index, data_before, color='red', label='BEFORE', alpha=0.5)
        ax.plot(data_after.index, data_after, color='black', label='AFTER', alpha=1)
        # sns.lineplot(x=data_before.index, y=data_before, color='red', label='BEFORE', ax=ax, alpha=0.5)
        # sns.lineplot(x=data_after.index, y=data_after, color='black', label='AFTER', ax=ax, linestyle="-", alpha=1)
        plt.legend()
        f.tight_layout()
        plt.show()

    def prediction_byIsolationForest(self, df, col_name, scaler_type='standardize', show = True):
        data = df[col_name].copy()

        # 인풋데이터 표준화
        if scaler_type == "standardize":
            scaler = StandardScaler()

        elif scaler_type == 'normalize':
            scaler = MinMaxScaler()
        else:
            raise Errors.CheckValue_Scaler

        data_reshaped = data.values.reshape(-1, 1)
        data_scaled = scaler.fit_transform(data_reshaped)

        if scaler_type == 'normalize':
            data_scaled = data_scaled.round(2)

        # print(f"data_scaled = {data_scaled}")
        # exit()
        # 모델 학습
        model_IF = IsolationForest()
        model_IF.fit(data_scaled)

        # 모델로 예측
        prediction = model_IF.predict(data_scaled)

        if show is True:
            df['OUTLIER'] = prediction
            # 시각화 : 전체
            fig, ax = plt.subplots(figsize=(10, 6))
            df_outlier = df.loc[df['OUTLIER'] == -1, ['MELT_WEIGHT']]
            print(len(df))
            print(len(df_outlier))
            ax.plot(df.index, df['MELT_WEIGHT'], color='black', label='Normal')
            ax.scatter(df_outlier.index, df_outlier, color='red', label = 'Outlier')
            fig.tight_layout()
            plt.legend()
            plt.show()


        """
        # fig, ax = plt.subplots(figsize=(10, 6))
        # 탐지결과 시각화 - 부분
        # month, day, hour = (3, 22, 22)
        # date = datetime.datetime(2020, month, day).date()
        # # date = datetime.datetime(2020, month, day).date()
        # df_show = df.loc[(df['DATE'] == date) & df['HOUR'] == hour]
        # 
        # a = df_show.loc[df['OUTLIER'] == -1, ['MELT_WEIGHT']]
        # print(len(df))
        # print(len(a))
        # ax.plot(df_show.index, df_show['MELT_WEIGHT'], color='black', label='Normal')
        # ax.scatter(a.index, a, color='red', label = 'Outlier')
        # fig.tight_layout()
        # plt.legend()
        # plt.show()
        """

        return prediction
    def correct_outliers_withStandardization(self, df, col_name):

        """

        :return:
        """


        # 다음행과 차이값 계산 (마지막 행은 NaN이 되므로 NaN은 0으로 처리)

        colName_diffPrev = f"{col_name}_DIFF(PREV)"


        df[colName_diffPrev] = df[col_name].diff(periods=1).abs()
        # df[colName_diffPrev] = df[col_name].diff(periods=1).abs()
        df[colName_diffPrev].fillna(0, inplace=True)
        # print(self.narrow_dataframe(df, [col_name, colName_diffPrev]))
        # exit()

        df['OUTLIER'] = self.prediction_byIsolationForest(df, col_name, scaler_type='standardize',show=True)
        # df['OUTLIER'] = self.prediction_byIsolationForest(df, colName_diffPrev, scaler_type='standardize',show=True)
        # df['OUTLIER'] = self.prediction_byIsolationForest(df, diffCol_name, scaler_type='normalize',show=True)
        # print(self.narrow_dataframe(df, ['MELT_WEIGHT', diffCol_name, 'OUTLIER']))

        # df['WEIGHT_DIFFNEXT'] = df['WEIGHT_NEXT'].sub(df['MELT_WEIGHT']).abs()
        # df['SPIKE'] = 0
        # df.loc[(df['WEIGHT_DIFFPREV'] > df['MELT_WEIGHT']) & (df['WEIGHT_DIFFNEXT'] > df['MELT_WEIGHT']), ]
        # df['WEIGHT_DIFFCOMB'] = df['WEIGHT_DIFFNEXT'].abs()+df['WEIGHT_DIFFPREV'].abs()
        # df.drop(columns=['WEIGHT_NEXT'], inplace=True)
        return df

    def fill_withFfill(self, df, col_name):
        """
        특정 칼럼의 NaN값을 'ffill' 방법으로 채워넣음
        :param df: NaN이 포함된 데이터 프레임
        :param col_name: NaN이 포함된 칼럼이름 (STR)
        :return: NaN이 처리된 데이터 프레임
        """

        # NaN은 ffill 처리
        df[col_name].fillna(method='ffill', inplace=True)

        # 예외처리. 데이터 프레임의 첫번째 데이터가 NaN이 되었을 경우, 원본 데이터 불러오기
        if np.isnan(df.at[0, col_name]):
            df.at[0, col_name] = self.df_org.at[0, col_name]

        output = df[col_name]

        return output

    def correct_outliers_overValue(self, df, col_name, cond_value):
        """
        주어진 칼럼과 값에 따라, 조건에 부합하는 값과 Outlier 칼럼 내 값을 np.nan, 1로 각 각 전환
        :return: 아웃라이어가 처리된 데이터 프레임
        """
        df['OUTLIER_OVER'] = 0

        df.loc[df[col_name] < cond_value, ['MELT_WEIGHT', 'OUTLIER_OVER']] = [np.nan, 1]
        df[col_name] = self.fill_withFfill(df, col_name)
        return df

    def correct_outliers_withThymeBoost(self, df, col_name, newCol_name, show=False):

        data = df[col_name]

        data_org = df[col_name].copy() # 차트 보여주기 위해 원본 저장

        # ThymeBoost(이하 TB) 불러오기
        boosted_model = tb.ThymeBoost()

        # TB로 아웃라이어 탐지
        output = boosted_model.detect_outliers(data)

        # True/False로 기록된 아웃라이어 여부를 0, 1로 치환하여 OUTLIER_TB 칼럼 생성 후 저장
        colName_outliers = f"{col_name}"
        df[newCol_name] = output['outliers']
        df[newCol_name].replace({True:1, False:0}, inplace=True)

        # TB 결과를 시각화하고싶다면
        if show is True:
            boosted_model.plot_results(output)
            boosted_model.plot_components(output)

        # 아웃라이어인 값은 np.NaN으로 변환
        df.loc[df[newCol_name] == 1, col_name] = np.nan

        df[col_name] = self.fill_withFfill(df, col_name)

        if show is True:
            self.plot_before_after(data_org, df[col_name])
            # f, ax = plt.subplots(1, 1, figsize=(20, 10))
            # sns.lineplot(x=df.index, y=data_org, color='yellow', label='BEFORE', ax=ax, alpha=0.5)
            # sns.lineplot(x = df.index, y=df[col_name], color='black', label ='AFTER(MEAN)', ax=ax, linestyle="-", alpha=1)
            # plt.legend()
            # f.tight_layout()
            # plt.show()

        return df

    def correct_outliers_withSpikeRemoval(self, df, col_name, newCol_name, sigma, show=False):

        data_org = df[col_name].copy()

        colName_diffPrev = f"(DIFFPREV)"
        colName_diffNext = f"(DIFFNEXT)"
        colName_diffTotal = f"(DIFFTOTAL)"
        df[newCol_name] = 0

        # 전행과 차이값 칼럼 추가
        df[colName_diffPrev] = df[col_name].diff(periods=1).pow(2)
        df[colName_diffNext] = df[colName_diffPrev].shift(-1)
        df[colName_diffTotal] = df[[colName_diffPrev, colName_diffNext]].sum(axis=1)

        # Shift로 NaN이 생긴 칼럼은 0으로 대체
        df.fillna(0, inplace=True)

        # 표준화값을 구한 후 새로운 칼럼 생성 후 값 저장

        df['(STD)'] = (df[colName_diffTotal] -  df[colName_diffTotal].mean())/df[colName_diffTotal].std()

        # 주어진 시그마를 넘는 값은 np.NaN으로 변환
        df.loc[df['(STD)'].abs() >= sigma, [col_name, newCol_name] ] = [np.nan, 1]
        df[col_name] = self.fill_withFfill(df, col_name)
        df.drop(columns=[colName_diffPrev, colName_diffNext, colName_diffTotal, '(STD)'], inplace=True)

        if show is True:
            self.plot_before_after(data_org, df[col_name])
            # f, ax = plt.subplots(1, 1, figsize=(20, 10))
            # sns.lineplot(x=df.index, y=data_org, color='red', label='BEFORE', ax=ax, alpha=0.5)
            # sns.lineplot(x=df.index, y=df[col_name], color='black', label='AFTER(MEAN)', ax=ax, linestyle="-",alpha=1)
            # plt.legend()
            # f.tight_layout()
            # plt.show()

        return df
    """
    한방향 버전
    def correct_outliers_withSpikeRemoval(self, df, col_name, newCol_name, sigma, show=False):

        data_org = df[col_name].copy()

        colName_diffPrev = f"(DIFFPREV)"
        df[newCol_name] = 0

        # 전행과 차이값 칼럼 추가
        df[colName_diffPrev] = df[col_name].diff(periods=1).pow(2)
        # Shift로 NaN이 생긴 칼럼은 0으로 대체
        df.fillna(0, inplace=True)

        # 표준화값을 구한 후 새로운 칼럼 생성 후 값 저장
        df['(STD)'] = (df[colName_diffPrev] -  df[colName_diffPrev].mean())/df[colName_diffPrev].std()

        # 주어진 시그마를 넘는 값은 np.NaN으로 변환
        df.loc[df['(STD)'].abs() >= sigma, [col_name, newCol_name] ] = [np.nan, 1]
        df[col_name] = self.fill_withFfill(df, col_name)
        df.drop(columns=['(DIFFPREV)', '(STD)'], inplace=True)

        if show is True:
            self.plot_before_after(data_org, df[col_name])

        return df
    """


    def correct_outliers_withMovingAverage(self, df, col_name, newCol_name, interval, sigma, show=False):
        data_org = df[col_name].copy()

        df[newCol_name] = 0
        # df['(MA)'] = df[col_name].rolling(interval, min_periods=1).sum()
        df['(MA)'] = df[col_name].rolling(interval, min_periods=1, center=True).mean()
        df['(MA-MSE)'] = (df[col_name] - df['(MA)']).pow(2)
        df['(MA-MSE)'] = (df['(MA-MSE)'] - df['(MA-MSE)'].mean())/df['(MA-MSE)'].std()

        df.loc[df['(MA-MSE)'].abs() >= sigma, [col_name, newCol_name]] = [np.nan, 1]
        df[col_name] = self.fill_withFfill(df, col_name)
        df.drop(columns = ['(MA)', '(MA-MSE)'], inplace=True)

        if show is True:
            self.plot_before_after(data_before= data_org, data_after= df[col_name])

        # print(self.narrow_dataframe(df, [col_name, '(MA)', '(MA-MSE)']))
        # print(df['(MA-MSE)'].describe().T)
        # sns.kdeplot(df['(MA-MSE)'])
        # plt.show()

        return df

    def correct_outliers_withMovingMedian(self, df, col_name, newCol_name, interval, sigma, show=False):
        data_org = df[col_name].copy()

        df[newCol_name] = 0
        # df['(MA)'] = df[col_name].rolling(interval, min_periods=1).sum()
        df['(MM)'] = df[col_name].rolling(interval, min_periods=1, center=True).median()
        # df['(MA)'] = df[col_name].rolling(interval, min_periods=1, center=True).mean()
        df['(MM-MSE)'] = (df[col_name] - df['(MM)']).pow(2)
        df['(MM-MSE)'] = (df['(MM-MSE)'] - df['(MM-MSE)'].mean())/df['(MM-MSE)'].std()

        df.loc[df['(MM-MSE)'].abs() >= sigma, [col_name, newCol_name]] = [np.nan, 1]
        df[col_name] = self.fill_withFfill(df, col_name)
        print(self.narrow_dataframe(df, [col_name, '(MM)']))
        df.drop(columns = ['(MM)', '(MM-MSE)'], inplace=True)

        if show is True:
            self.plot_before_after(data_before= data_org, data_after= df[col_name])

        # print(self.narrow_dataframe(df, [col_name, '(MA)', '(MA-MSE)']))
        # print(df['(MA-MSE)'].describe().T)
        # sns.kdeplot(df['(MA-MSE)'])
        # plt.show()

        return df

    # def correct_outliers_withProphet(self, df , col_name):
    #     data = df[['DATE_TIME', col_name]].copy()
    #     data.rename(columns = {'DATE_TIME':'ds', col_name:'y'}, inplace=True)
    #
    #     m = Prophet(changepoint_range=0.95)



    def DATAFRAME_OUTLIERREMOVED(self):
        data_before = self.df_prcd['MELT_WEIGHT'].copy()
        df = self.df_prcd
        # df = self.correct_outliers_overValue(df, 'MELT_WEIGHT', 3000)
        df = self.correct_outliers_withThymeBoost(df= df, col_name='MELT_WEIGHT', newCol_name= 'OUTLIER_WGT(TB)', show= False)

        # df = self.correct_outliers_withSpikeRemoval(df= df, col_name='MELT_WEIGHT', newCol_name='OUTLIER_WTG(SPK)', sigma= 6, show=False)
        for interval in [60, 30, 15, 9, 3]:
            df = self.correct_outliers_withMovingMedian(df = df, col_name= 'MELT_WEIGHT', newCol_name = f'OUTLIER_WTG(MM_{interval})', interval=interval, sigma=5, show=False)

        # df = self.correct_outliers_withMovingAverage(df = df, col_name= 'MELT_WEIGHT', newCol_name = 'OUTLIER_WTG(MA)', interval=3, sigma=3, show=False)

        print(df[df['MELT_WEIGHT'].isnull()]['MELT_WEIGHT'])
        print(df)
        self.plot_before_after(data_before, df['MELT_WEIGHT'])

        return df

dc = DataCleaning()

df = dc.DATAFRAME_OUTLIERREMOVED()
