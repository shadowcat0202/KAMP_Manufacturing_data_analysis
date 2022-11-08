import pandas as pd
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns
import tsfresh.examples

from DataPreprocess import DataPreprocess

if __name__ == '__main__':

    # 데이터프레임 전처리용 클래스
    dp = DataPreprocess()
    #
    # # 전처리된 데이터프레임
    df = dp.df_prcd

    print(df)

    df.dropna(axis=0, inplace=True)
    x = df[['DATE', 'DATE_TIME', 'MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP']]
    y = df['OK']

    # print(df['NUM'].value_counts())
    # print(x.value_counts())
    # print(y.value_counts())

    from tsfresh import extract_features
    from tsfresh import extract_relevant_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh import select_features
    from multiprocessing import freeze_support

    # print(df.dtypes)
    freeze_support()

    # print(df.index.tolist())
    # features = extract_relevant_features(x, y, column_id="NUM", column_sort="DATE_TIME")
    # features = extract_features(x, column_id="NUM", column_sort="DATE_TIME")
    features = extract_features(x, column_id="DATE", column_sort="DATE_TIME")
    print(f"FEATURES.info = \n{features.info}")
    print(f"FEATURES.shape = \n{features.shape}")
    print(f"FEATURES.columns = \n{features.columns}")
    print(f"FEATURES = \n{features}")
    #
    impute(features)
    filtered_featrues = select_features(features, y)
    print(f"FILTERED_FEATURES = \n{filtered_featrues}")