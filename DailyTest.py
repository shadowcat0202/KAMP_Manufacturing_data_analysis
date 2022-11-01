import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DailyVyDPP import DailyVyDPP

pd.set_option("display.max_columns", None)


if __name__ == "__main__":
    df = pd.read_csv("../data/melt_weight_outlier/weight_preprocess_1.csv")
    origin_df = pd.read_csv("../data/dataset.csv")
    # plt.plot(df["MELT_WEIGHT"])
    # plt.plot(origin_df["MELT_WEIGHT"])
    # plt.show()

    # print(df[(df["MELT_WEIGHT"] == origin_df["MELT_WEIGHT"]) == False]["MELT_WEIGHT"])
    # print(origin_df[(df["MELT_WEIGHT"] == origin_df["MELT_WEIGHT"]) == False]["MELT_WEIGHT"])

    print(df["MELT_WEIGHT"].describe())
    print(origin_df["MELT_WEIGHT"].describe())