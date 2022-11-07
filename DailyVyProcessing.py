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

origin_df = pd.read_csv("../data/dataset.csv")

outlier_1 = pd.read_csv("../data/melt_weight_outlier/weight_preprocess_1.csv")

outlier_2_path = "../data/melt_weight_outlier/weight_preprocess_2.csv"
outlier_2 = pd.read_csv(outlier_2_path)

outlier_3_path = "../data/melt_weight_outlier/weight_preprocess_3.csv"
outlier_3 = pd.read_csv(outlier_3_path)



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
    # plt.plot(my_df_weight.iloc[820000:821000])
    # plt.plot(outlier_2.loc[820000:821000, "MELT_WEIGHT"])
    # plt.show()

###########################################################################################

    #################### 2차 Outlier ######################################

    # print(outlier_2.info())
    # print(outlier_2.describe())

    # plt.plot(outlier_2["MELT_WEIGHT"])
    # plt.show()

    # weight_outlier = outlier_2["MELT_WEIGHT"]
    # weight_outlier_part1 = weight_outlier[:20000]
    # plt.plot(origin_df.loc[:20000, "MELT_WEIGHT"], label="origin")
    # plt.plot(weight_outlier_part1, label="1st processing")
    # plt.show()

    ################### 앞의 값과 비교해서 많이 클 경우 앞, 뒤 평균 값을 넣자 #######################
    # for i in range(1, 20000):
    #     try:
    #         devide_value = weight_outlier_part1[i] / weight_outlier_part1[i-1]
    #     except (ZeroDivisionError, IndexError) as e:
    #         continue
    #
    #     if devide_value >= 1.5 and (weight_outlier_part1[i] // 100 != 0):
    #         try:
    #             weight_outlier_part1[i] = (weight_outlier_part1[i-1] + weight_outlier_part1[i+1]) // 2
    #         except IndexError as e:
    #             continue
    # plt.plot(weight_outlier_part1[:20000], label="after processing")
    # plt.legend(loc="best")
    # plt.show()

    ### 함수로 해보자구여
    # obj = DailyVyDPP(outlier_2)

    ## test
    # for i in range(0, len(outlier_2), 20000):
    #     try:
    #         obj.melt_weight_outlier_soft_increase(i+1, i+20000)
    #     except (ValueError, KeyError) as e:
    #         obj.melt_weight_outlier_soft_increase(i+1, len(outlier_2))

    # plt.plot(origin_df["MELT_WEIGHT"], label="original")
    # obj.melt_weight_outlier_soft_increase(1, len(outlier_2)) # outlier_2 에 저장됨

    # plt.plot(outlier_2["MELT_WEIGHT"], label="processing")
    # plt.legend(loc="best")
    # plt.show()

    # outlier_2.to_csv("../data/melt_weight_outlier/weight_preprocess_3.csv")

#######################################################################################
    # 4000 이상 값은 np.nan 으로 대체할거야....

    # plt.plot(origin_df["MELT_WEIGHT"], label="original")
    # plt.plot(outlier_2["MELT_WEIGHT"], label= "1st process")
    # # plt.plot(outlier_3["MELT_WEIGHT"], label="2nd process")
    # # plt.show()
    # print(outlier_3.info())
    # print(f"4000 이하의 갯수 : {len(outlier_3) - (outlier_3[outlier_3['MELT_WEIGHT'] >= 4000]['MELT_WEIGHT'].count())}") # 834948
    #
    # # 4000 이상 값을 NaN 으로 대체
    # outlier_3["MELT_WEIGHT"] = outlier_3["MELT_WEIGHT"].apply(lambda x: np.nan if x >= 4000 else x)
    # print(outlier_3.info())
    #
    # # NaN 값을 앞의 값으로 채워줌
    # outlier_3 = outlier_3.fillna(method="ffill") # 앞의 값으로 채워줌
    # print(outlier_3.info())
    #
    # plt.plot(outlier_3["MELT_WEIGHT"], label="4000 ffill", linestyle="-")


    # 2000 이상 값을 NaN 으로 대체
    # outlier_3["MELT_WEIGHT"] = outlier_3["MELT_WEIGHT"].apply(lambda x: np.nan if x >= 2300 else x)
    # outlier_3 = outlier_3.fillna(method="ffill")

    # plt.plot(outlier_3["MELT_WEIGHT"], label="2300 ffill", linestyle="-")

    # outlier_3.to_csv("../data/melt_weight_outlier/after2300ffill.csv")
    # plt.legend(loc="best")
    # plt.show()