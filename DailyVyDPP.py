import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DailyVyDPP:
    def __init__(self, df):
        self.df = df
        self.weight = df["MELT_WEIGHT"]
        self.speed = df["MOTORSPEED"]
        self.temp = df["MELT_TEMP"]

    def melt_weight_outlier(self, start, end):
        """
        1차적으로 rough 하게 outlier를 수정
        많이 튄 값을 10으로 나눠줌
        :param start:
        :param end:
        :return: outlier 수정 된 series 반환
        """

        # plt.plot(self.weight.iloc[start:end]) # 기존 그래프

        try:
            for i in range(start, end):
                devide_value = self.weight[i] / self.weight[i-1]

                if devide_value >= 8 and self.weight[i] >= 1000:
                    self.weight[i] //= 10
        except ZeroDivisionError as e:
            pass

        # plt.plot(self.weight.iloc[start:end])
        # plt.show()

        return self.weight

    def melt_weight_outlier_soft_increase(self, start, end):
        """
        1차에서 한번 필터링된 아웃라이어를 2차적으로 수정해줌
        :param start:
        :param end:
        :return:
        """
        # plt.plot(self.weight.iloc[start:end], label="1st processing")

        for i in range(start, end):
            try:
                devide_value = self.weight[i] / self.weight[i-1]
            except (ZeroDivisionError, IndexError, KeyError, ValueError) as e:
                continue

            if devide_value >= 1.5 and (self.weight[i] // 100 != 0):
                try:
                    self.weight[i] = (self.weight[i-1] + self.weight[i+1] // 2)
                except (IndexError, KeyError, ValueError) as e:
                    continue

        # plt.plot(self.weight.iloc[start:end], label="2nd processing")
        # plt.legend(loc="best")
        # plt.show()

        return self.weight


    def melt_weight_outlier_decline_rough(self, start, end):
        for i in range(start, end):
            dividend = self.weight[i] # 나눠지는 수, 앞의 값
            divisor = self.weight[i+1] # 나누는 수, 뒤의 값

            standard = 5



            if dividend < 100: # 100 미만으로는 감소율이 크므로 pass
                continue # 이 경우 다른 전처리를 해줘야 겠지

            if divisor == 0: # 0으로 나눌 수 없으니까
                divisor = 1

            divide_value = dividend / divisor

            if divide_value >= standard: # 나눈 값이 기준보다 크다면 ==> 기준은 수정 가능
                self.weight[i+1] = dividend # 뒤의 값을 앞의값으로 치환

        return self.weight

    def highVal_replace_nan(self, value):
        if value >= 4000:
            return np.nan

    # def returnDF(self):
    #     return self.df
