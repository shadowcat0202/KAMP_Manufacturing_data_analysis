import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataPreprocess import DataPreprocess
from DailyVyDPP import DailyVyDPP

pd.set_option("display.max_columns", None)

dataPreprocess = DataPreprocess()
df = dataPreprocess.processed_dataframe()
# print(df)

outlier_1 = pd.read_csv("../data/melt_weight_outlier/weight_preprocess_1.csv")
outlier_2 = pd.read_csv("../data/melt_weight_outlier/weight_preprocess_2.csv")




if __name__ == "__main__":
    my_df = df[["DATE_TIME", "MELT_TEMP", "MOTORSPEED", "MELT_WEIGHT", "INSP", "OK"]]
    my_df_weight = my_df["MELT_WEIGHT"].copy() # 원본
    # print(my_df.head(30))


############################################## outlier_1 ###########################

    # 1. 급증한 구간 전처리 1차 ==> ROUGH

    # dailyvy = DailyVyDPP(my_df)
    # dailyvy_weights = dailyvy.melt_weight_outlier(1, len(my_df))
    # plt.plot(my_df_weight)
    # plt.plot(my_df["MELT_WEIGHT"])
    # plt.show()

    # save_df = my_df.copy()
    # save_df["MELT_WEIGHT"] = dailyvy_weights

    # plt.plot(my_df_weight)
    # plt.plot(save_df["MELT_WEIGHT"])
    # plt.show()

    # save_df.to_csv("../data/melt_weight_outlier/weight_preprocess_1.csv", sep=",")

###########################################################################################
    # pd.set_option('display.max_rows', None)

    # print(outlier_1.describe())
    # print(outlier_1.loc[820000:820500, "MELT_WEIGHT"])
    # print(outlier_1.info())
    # print(outlier_1.describe())
    # print(outlier_1.loc[:10,"MELT_TEMP"])

    # plt.plot(my_df_weight.iloc[820000:820500])
    # plt.plot(outlier_1.loc[820000:820500, "MELT_WEIGHT"])
    # plt.show()

    # MELT_WEIGHT 가 0이 나오는 구간을 찾을거야
    # mask = outlier_1.loc[820000:821000, "MELT_WEIGHT"] == 0
    # zeroPoint = outlier_1[outlier_1["MELT_WEIGHT"] == 0]
    # zeroIdx = zeroPoint.loc[820000:821000, "MELT_WEIGHT"]
    # print(zero) # 820275, 820281, 820283, 820289, 820293, 820296, 820298, 820304
    """
    820500 이후에서 zero 가 안나오네, 1 또는 2인가 보다.
    """
    # onePoint = outlier_1[outlier_1["MELT_WEIGHT"] == 1]
    # oneIdx = onePoint.loc[820000:821000, "MELT_WEIGHT"]
    # print(oneIdx) # ~, 820834


    # print(outlier_1.loc[820500:821000, "MELT_WEIGHT"])
    # plt.plot(my_df_weight.iloc[820500:821500])
    # plt.plot(outlier_1.loc[820500:821500, "MELT_WEIGHT"])
    # plt.show()

    # 820500 ~ 821000 에서 max 값을 보자 ==> 820913
    # print(outlier_1[outlier_1.loc[820500:821000, "MELT_WEIGHT"].max()])
    # maxVal = outlier_1.loc[820500:821000, "MELT_WEIGHT"].max()
    # print(outlier_1[outlier_1["MELT_WEIGHT"] == maxVal].loc[820500:821000, "MELT_WEIGHT"]) # 820913

    # 820834 ~ 820913 값을 복사해서 붙여넣어봅시다.
    # print(820913 - 820834) # 79
    # 이상한 구간은 820276 부터 시작, 820396 까지 값이 이상 (증가해야하는데 증가안함)
    # 820317 ~ 820396을 820834 ~ 820913 값을 복사하고
    # 820276 ~ 820316 구간은 0으로 지정
    # outlier_1.loc[820276:820316, "MELT_WEIGHT"] = 0
    # for i in range(820317, 820397):
    #     outlier_1.loc[i, "MELT_WEIGHT"] = outlier_1.loc[i+517, "MELT_WEIGHT"]

    # print(outlier_1.loc[820276:821000, "MELT_WEIGHT"])

    # plt.plot(my_df_weight.iloc[820000:821500])
    # plt.plot(outlier_1.loc[820000:821500, "MELT_WEIGHT"])
    # plt.show()

    # my_df["MELT_WEIGHT"] = outlier_1["MELT_WEIGHT"]

    # plt.plot(my_df.loc[820000:821500, "MELT_WEIGHT"])
    # plt.plot(outlier_1.loc[820000:821500, "MELT_WEIGHT"])
    # plt.show()
    # print(my_df.loc[820000:820300,"MELT_WEIGHT"]) # 820265 : 167 (이전 값 17, 이후값 15)
    # my_df.loc[820265,"MELT_WEIGHT"] = 16 # 16으로 값 바꿔줌

    # my_df.to_csv("../data/melt_weight_outlier/weight_preprocess_2.csv", sep=",")
###########################################################################################
    plt.plot(my_df_weight.iloc[820000:821000])
    plt.plot(outlier_2.loc[820000:821000, "MELT_WEIGHT"])
    plt.show()