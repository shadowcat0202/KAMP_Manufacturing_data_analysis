import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataPreprocess import DataPreprocess
from DailyVyDPP import DailyVyDPP

pd.set_option("display.max_columns", None)

origin_df = pd.read_csv("../data/dataset.csv")
outlier_2_path = "../data/melt_weight_outlier/weight_preprocess_2.csv"
outlier_2 = pd.read_csv(outlier_2_path)
outlier_3_path = "../data/melt_weight_outlier/weight_preprocess_3.csv"
outlier_3 = pd.read_csv(outlier_3_path)

if __name__ == "__main__":
    print(outlier_2.info())
    print(outlier_2.describe())

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
    obj = DailyVyDPP(outlier_2)

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
    # plt.plot(outlier_3["MELT_WEIGHT"], label="2nd process")
    # plt.show()
    print(outlier_3.info())
    print(f"4000 이하의 갯수 : {len(outlier_3) - (outlier_3[outlier_3['MELT_WEIGHT'] >= 4000]['MELT_WEIGHT'].count())}") # 834948

    # 4000 이상 값을 NaN 으로 대체
    outlier_3["MELT_WEIGHT"] = outlier_3["MELT_WEIGHT"].apply(lambda x: np.nan if x >= 4000 else x)
    print(outlier_3.info())

    # NaN 값을 앞의 값으로 채워줌
    outlier_3 = outlier_3.fillna(method="ffill") # 앞의 값으로 채워줌
    print(outlier_3.info())

    # plt.plot(outlier_3["MELT_WEIGHT"], label="4000 ffill", linestyle="--")


    # 2000 이상 값을 NaN 으로 대체
    outlier_3["MELT_WEIGHT"] = outlier_3["MELT_WEIGHT"].apply(lambda x: np.nan if x >= 2000 else x)
    outlier_3 = outlier_3.fillna(method="ffill")

    # plt.plot(outlier_3["MELT_WEIGHT"], label="2000 ffill", linestyle=":")


    # plt.legend(loc="best")
    # plt.show()