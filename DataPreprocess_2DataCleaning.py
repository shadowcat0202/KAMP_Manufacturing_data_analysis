import numpy as np
from DataPreProcess_1DataLoad import DataLoad

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


class DataCleaning(DataLoad):
    def __init__(self):
        super().__init__()
        self.df_clnd = self.DATAFRAME_OUTLIERREMOVED(show=False)

    def narrow_dataframe(self, df, cols):
        """
        데이터프레임에서 cols에 입력된 부분으로 추려줌
        :param df: 데이터프레임
        :param cols: 지정하고 싶은 칼럼 (STR or LIST)
        :return: 지정된 칼럼만 있는 데이터 프레임
        """
        return df[cols]

    def plot_before_after(self, data_before, data_after, figure_name='Figure'):
        """
        두 개의 시리즈를 line plot 형태로 시각화
        :param data_before: 비교 대상 데이터프레임
        :param data_after: 비교 대상 데이터 프레임
        :param figure_name: 그래프의 타이틀
        :return: None
        """
        f, ax = plt.subplots(1, 1, figsize=(18, 9))
        ax.plot(data_before.index, data_before, color='red', label='BEFORE', alpha=0.5)
        ax.plot(data_after.index, data_after, color='black', label='AFTER', alpha=1)
        # sns.lineplot(x=data_before.index, y=data_before, color='red', label='BEFORE', ax=ax, alpha=0.5)
        # sns.lineplot(x=data_after.index, y=data_after, color='black', label='AFTER', ax=ax, linestyle="-", alpha=1)
        plt.legend()
        f.tight_layout()
        f.suptitle(figure_name)
        plt.show()

    def fill_withFfill(self, df, col_name):
        """
        특정 칼럼의 NaN값을 'ffill' 방법으로 채워넣음
        :param df: NaN이 포함된 데이터 프레임
        :param col_name: NaN이 포함된 칼럼이름 (STR)
        :return: NaN이 처리된 데이터 프레임
        """

        # NaN은 ffill 처리
        df[col_name].fillna(method='ffill', inplace=True)
        # print(df.at[0, col_name])

        # 예외처리. 데이터 프레임의 첫번째 데이터가 NaN이 되었을 경우, 원본 데이터 불러오기
        if np.isnan(df.at[0, col_name]):
            df.at[0, col_name] = self.df_org.at[0, col_name]

        output = df[col_name]

        return output

    def correct_outliers_withThymeBoost(self, df, col_name, newCol_name, show=False):
        """
        ThymeBoost를 이용하여 아웃라이어 탐지. 이후 아웃라이어는 np.NaN 변환 및 칼럼 생성 후 1표기 (아닐경우 0). 이후 fillna
        :param df:
        :param col_name: 아웃라이어가 들어있는 칼럼의 이름 (STR only)
        :param newCol_name: 아웃라이어를 0, 1로 표기할 새로운 칼럼의 이름
        :param show: 처리 전후를 플랏할 것인지 여부
        :return: ThymeBoost로 아웃라이어가 처리된 데이터 프레임
        """

        # 아웃라이어가 들어있는 시리즈
        data = df[col_name]

        # 시리즈 원본 저장 (데이터를 플랏으로 처리 전후 차이를비교하기 위해)
        data_org = df[col_name].copy()

        # ThymeBoost(이하 TB) 불러오기
        boosted_model = tb.ThymeBoost()

        # TB로 아웃라이어 탐지
        output = boosted_model.detect_outliers(data)

        # True/False로 기록된 아웃라이어 여부를 0, 1로 치환하여 OUTLIER_TB 칼럼 생성 후 저장
        colName_outliers = f"{col_name}"
        df[newCol_name] = output['outliers']
        df[newCol_name].replace({True: 1, False: 0}, inplace=True)

        # TB 결과를 시각화하고싶다면
        if show is True:
            boosted_model.plot_results(output)
            boosted_model.plot_components(output)

        # 아웃라이어인 값은 np.NaN으로 변환 이후 fillna
        df.loc[df[newCol_name] == 1, col_name] = np.nan
        df[col_name] = self.fill_withFfill(df, col_name)

        # 처리 전/후 시각화
        if show is True:
            fig_title = "THYMEBOOST (BEFORE/AFTER)"
            self.plot_before_after(data_org, df[col_name], figure_name=fig_title)

        return df

    def correct_outliers_withSpikeRemoval(self, df, col_name, newCol_name, sigma, show=False):

        # 시리즈 원본 저장 (데이터를 플랏으로 처리 전후 차이를비교하기 위해)
        data_org = df[col_name].copy()

        # 새로운 칼럼들의 이름 생성
        colName_diffPrev = f"(DIFFPREV)"  # 이전 행의 값과 차이
        colName_diffNext = f"(DIFFNEXT)"  # 다음 행의 값과 차이
        colName_diffTotal = f"(DIFFTOTAL)"  # 이전행/다음행 값의 합

        # OUTLIER 0, 1로 표기할 신규 칼럼 생성. 기본값 0
        df[newCol_name] = 0

        # 전행과 차이값 칼럼 추가
        df[colName_diffPrev] = df[col_name].diff(periods=1).pow(2)  # 이전값과 차이 제곱 (큰 차이를 더 극대화하기 위해) 칼럼생성
        df[colName_diffNext] = df[colName_diffPrev].shift(-1)  # 다음값과 차이 제곱 칼럼 생성
        df[colName_diffTotal] = df[[colName_diffPrev, colName_diffNext]].sum(axis=1)  # 이전값 차이 제곱과 다음값 차이 제곱을 더한 칼럼 생성

        # Shift로 NaN이 생긴 칼럼은 0으로 대체
        df.fillna(0, inplace=True)

        # 표준화값을 구한 후 새로운 칼럼(STD) 생성 후 값 저장
        df['(STD)'] = (df[colName_diffTotal] - df[colName_diffTotal].mean()) / df[colName_diffTotal].std()

        # 주어진 시그마를 넘는 값은 np.NaN으로 변환. 이후 ffill 처리
        df.loc[df['(STD)'].abs() >= sigma, [col_name, newCol_name]] = [np.nan, 1]
        df[col_name] = self.fill_withFfill(df, col_name)

        # 임시로 생성한 칼럼들은 삭제
        df.drop(columns=[colName_diffPrev, colName_diffNext, colName_diffTotal, '(STD)'], inplace=True)

        # 처리 전/후 시각화
        if show is True:
            fig_title = f"Spike Removal[{col_name}] (sigma {sigma}) Before/After"
            self.plot_before_after(data_org, df[col_name])

        return df

    def correct_outliers_withMovingAverage(self, df, col_name, newCol_name, interval, sigma, show=False):
        # 시리즈 원본 저장 (데이터를 플랏으로 처리 전후 차이를비교하기 위해)
        data_org = df[col_name].copy()

        # OUTLIER 0, 1로 표기할 신규 칼럼 생성. 기본값 0
        df[newCol_name] = 0

        # 주어진 조건의 윈도우 생성. 이동평균값과 관측값의 차이를 제곱. 이후 표준화처리
        df['(MA)'] = df[col_name].rolling(interval, min_periods=1, center=True).mean()  # 윈도우 생성 (관측값이 중앙으로)
        df['(MA-MSE)'] = (df[col_name] - df['(MA)']).pow(2)  # 이동평균과 관측값의 차이를 제곱
        df['(MA-MSE)'] = (df['(MA-MSE)'] - df['(MA-MSE)'].mean()) / df['(MA-MSE)'].std()  # 차이값의 제곱을 표준화

        # 주어진 시그마 이상의 값은 NaN으로 처리. 이후 ffill 처리
        df.loc[df['(MA-MSE)'].abs() >= sigma, [col_name, newCol_name]] = [np.nan, 1]  # 주어진 시그마 이상의 표준화된 값은 NaN처리
        df[col_name] = self.fill_withFfill(df, col_name)  # NaN을 ffill 처리

        # 임시로 생성한 칼럼들 드랍
        df.drop(columns=['(MA)', '(MA-MSE)'], inplace=True)

        # 처리 전/후 시각화
        if show is True:
            fig_title = f"Moving Average[{col_name}] (interval {interval}, sigma {sigma}) Before/After"
            self.plot_before_after(data_before=data_org, data_after=df[col_name])

        return df

    def correct_outliers_withMovingAverageLog(self, df, col_name, newCol_name, interval, sigma, show=False):
        # 시리즈 원본 저장 (데이터를 플랏으로 처리 전후 차이를비교하기 위해)
        data_org = df[col_name].copy()

        # OUTLIER 0, 1로 표기할 신규 칼럼 생성. 기본값 0
        df[newCol_name] = 0

        df[col_name] = np.where(df[col_name] == 0, 0, np.log10(df[col_name]))

        # 주어진 조건의 윈도우 생성. 이동평균값과 관측값의 차이를 제곱. 이후 표준화처리
        df['(MA)'] = df[col_name].rolling(interval, min_periods=1, center=True).mean()  # 윈도우 생성 (관측값이 중앙으로)
        df['(MA-MSE)'] = (df[col_name] - df['(MA)'])  # 이동평균과 관측값의 차이를 제곱
        # df['(MA-MSE)'] = (df[col_name] - df['(MA)']).pow(2) # 이동평균과 관측값의 차이를 제곱
        df['(MA-MSE)'] = (df['(MA-MSE)'] - df['(MA-MSE)'].mean()) / df['(MA-MSE)'].std()  # 차이값의 제곱을 표준화

        # 주어진 시그마 이상의 값은 NaN으로 처리. 이후 ffill 처리
        df.loc[df['(MA-MSE)'].abs() >= sigma, [col_name, newCol_name]] = [np.nan, 1]  # 주어진 시그마 이상의 표준화된 값은 NaN처리
        df[col_name] = self.fill_withFfill(df, col_name)  # NaN을 ffill 처리

        df[col_name] = 10 ** df[col_name]
        # print(df[col_name])
        # 임시로 생성한 칼럼들 드랍
        df.drop(columns=['(MA)', '(MA-MSE)'], inplace=True)

        # 처리 전/후 시각화
        if show is True:
            fig_title = f"Moving Average[{col_name}] (interval {interval}, sigma {sigma}) Before/After"
            self.plot_before_after(data_before=data_org, data_after=df[col_name])

        return df

    def correct_outliers_withMovingMedian(self, df, col_name, newCol_name, interval, sigma, show=False):
        data_org = df[col_name].copy()

        # OUTLIER 0, 1로 표기할 신규 칼럼 생성. 기본값 0
        df[newCol_name] = 0

        # 주어진 조건의 윈도우 생성. 이동중앙값과 관측값의 차이를 제곱. 이후 표준화처리
        df['(MM)'] = df[col_name].rolling(interval, min_periods=1, center=True).median()  # 윈도우 생성 (관측값이 중앙으로)
        df['(MM-MSE)'] = (df[col_name] - df['(MM)']).pow(2)  # 이동중앙값과 관측값의 차이를 제곱
        df['(MM-MSE)'] = (df['(MM-MSE)'] - df['(MM-MSE)'].mean()) / df['(MM-MSE)'].std()  # 차이값의 제곱을 표준화

        df.loc[df['(MM-MSE)'].abs() >= sigma, [col_name, newCol_name]] = [np.nan, 1]  # 주어진 시그마 이상의 표준화된 값은 NaN처리
        df[col_name] = self.fill_withFfill(df, col_name)  # NaN을 ffill 처리

        # 임시로 생성한 칼럼들 드랍
        df.drop(columns=['(MM)', '(MM-MSE)'], inplace=True)

        # 처리 전/후 시각화
        if show is True:
            fig_title = f"Moving Median[{col_name}] (interval {interval}, sigma {sigma}) Before/After"
            self.plot_before_after(data_before=data_org, data_after=df[col_name], figure_name=fig_title)

        return df

    def DATAFRAME_OUTLIERREMOVED(self, show=False):
        data_before = self.df_org['MELT_WEIGHT'].copy()
        df = self.df_org.copy()
        df = self.correct_outliers_withThymeBoost(df=df, col_name='MELT_WEIGHT', newCol_name='OUTLIER_WGT(TB)',
                                                  show=False)

        for interval in [60, 30, 15, 9, 3]:
            df = self.correct_outliers_withMovingMedian(df=df, col_name='MELT_WEIGHT',
                                                        newCol_name=f'OUTLIER_WTG(MM_{interval})', interval=interval,
                                                        sigma=5, show=False)

        # 처리 전/후 시각화
        if show is True:
            self.plot_before_after(data_before, df['MELT_WEIGHT'], figure_name="Data Cleaning Before/After")

        return df
