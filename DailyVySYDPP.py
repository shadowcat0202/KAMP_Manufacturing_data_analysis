import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataPreprocess import DataPreprocess
from DailyVyDPP import DailyVyDPP

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

origin_df = pd.read_csv("../data/dataset.csv")

outlier_1 = pd.read_csv("../data/melt_weight_outlier/weight_preprocess_1.csv")

outlier_2_path = "../data/melt_weight_outlier/weight_preprocess_2.csv"
outlier_2 = pd.read_csv(outlier_2_path)

outlier_3_path = "../data/melt_weight_outlier/weight_preprocess_3.csv"
outlier_3 = pd.read_csv(outlier_3_path)

outlier_4_path = "../data/melt_weight_outlier/after2300ffill.csv"
outlier_4 = pd.read_csv(outlier_4_path)

if __name__ == "__main__":

    start = 113000
    end = 130000

    # plt.plot(outlier_3["MELT_WEIGHT"])
    # plt.plot(outlier_4["MELT_WEIGHT"])
    # plt.plot(outlier_3.loc[start:end, "MELT_WEIGHT"], label="3rd preprocess")
    # plt.plot(outlier_4.loc[start:end, "MELT_WEIGHT"], label="2300 ffill")

    # plt.show()

    # for i in range(start, end):
    #     dividend = outlier_4.loc[i, "MELT_WEIGHT"] # 나눠지는 수, 앞의 값
    #     divisor = outlier_4.loc[i+1, "MELT_WEIGHT"] # 나누는 수, 뒤의 값
    #
    #     if dividend < 100: # 100 미만으로는 감소율이 크므로 pass
    #         continue
    #
    #     if divisor == 0: # 0으로 나눌 수 없으니까
    #         divisor = 1
    #
    #     divide_value = dividend / divisor
    #
    #     # if divide_value >= 10: # 나눈 값이 10보다 크다면
    #     # if divide_value >= 5: # 나눈 값이 5보다 크다면
    #     if divide_value >= 2: # 나눈 값이 .15보다 크다면
    #         outlier_4.loc[i+1, "MELT_WEIGHT"] = dividend # 뒤의 값을 앞의 값으로 치환
    #
    #
    #
    # plt.plot(outlier_4.loc[start:end, "MELT_WEIGHT"], label="decline cleaning 1st")
    # plt.legend(loc="best")
    # plt.show()
    #

    # print(outlier_4.loc[start:end, "MELT_WEIGHT"])

    compare_df = pd.DataFrame({
        "original_weight" : origin_df["MELT_WEIGHT"],
        "2300_ffill_weight" : outlier_4["MELT_WEIGHT"]
    })

    print(compare_df[start:end])