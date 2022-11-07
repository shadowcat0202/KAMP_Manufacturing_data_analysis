from DataPreProcess_1DataLoad import DataLoad

class NewFeatures():
    def __init__(self):
        self.ls_date = ['MONTH', 'WEEK', 'DATE', 'WEEKDAY', 'HOUR']

    def expand_date_columns(self, df):
        """
        주어진 데이터 프레임의 Date 칼럼의 데이터를 가지고 date 타입의 다양한 데이터 칼럼을 생성
        :param df: 확장할 데이터 프레임
        :return: 시간 데이터 칼럼이 추가된 데이터 프레임
        """
        df['MONTH'] = df['DATE_TIME'].dt.month
        df['WEEK'] = df['DATE_TIME'].dt.isocalendar().week
        df['DATE'] = df['DATE_TIME'].dt.date
        df['WEEKDAY'] = df['DATE_TIME'].dt.weekday  # 0 == Monday, 6 == Sunday
        df['HOUR'] = df['DATE_TIME'].dt.hour

        return df

    def calculate_change_columns(self, df, colName_from, colName_new):
        """
        주어진 데이터 프레임 내 칼럼의 행 간 변화량을 계산함 [val*100(%)]. 첫 행 NaN은 0으로 표기
        :param df:
        :param colName_from:
        :param colName_new:
        :return:
        """
        df[colName_new] = df[colName_from].pct_change(periods=1)
        df[colName_new].fillna(0, inplace=True)

        return df

    def generate_columns_withLagFeatures(self, df, col_feats, back_to):

        log = {}
        for col in col_feats:
            ls_cols = []
            ls_cols.append(col)
            for t in range(1, back_to+1):
                newCol_name = f"{col}(t-{t})"
                ls_cols.append(newCol_name)
                df[newCol_name] = df[col].shift(t)
            log[col] = ls_cols

        return df, log

    def generate_columns_withWindowFeatures(self, df, col_feats, window_size):

        methods = ['std', 'mean', 'max', 'min', 'median']
        log = {}
        ls_cols = []
        for col in col_feats:
            ls_cols.append(col)
            rolling = df[col].rolling(window_size)
            for mth in methods:
                newCol_name = f"{col}_{mth}({window_size})"
                if mth == 'mean':
                    df[newCol_name] = rolling.mean()
                elif mth == 'std':
                    df[newCol_name] = rolling.std()
                elif mth == 'median':
                    df[newCol_name] = rolling.median()
                elif mth == 'min':
                    df[newCol_name] = rolling.min()
                elif mth == 'max':
                    df[newCol_name] = rolling.max()
                ls_cols.append(newCol_name)
                log[col] = ls_cols

        return df, log

    """
    # df = df.iloc[573739:573826]
    # df = df[['MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP', 'OUTLIER_WGT(TB)',
    #          'OUTLIER_WTG(MM_60)', 'OUTLIER_WTG(MM_30)', 'OUTLIER_WTG(MM_15)',
    #          'OUTLIER_WTG(MM_9)', 'OUTLIER_WTG(MM_3)', 'DATE_TIME', 'OK', 'DATE']]
    # df['NUM'] = df.index.astype(int)
    # df['NUM'] = df['NUM'] % 10
    #
    #
    # # x = df[['NUM', 'MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP']]
    # x = df[['NUM', 'DATE_TIME', 'MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP']]
    # y = df['OK']
    #
    # print(df['NUM'].value_counts())
    # print(x.value_counts())
    # print(y.value_counts())
    #
    # from tsfresh import extract_features
    # from tsfresh import extract_relevant_features
    # from tsfresh.utilities.dataframe_functions import impute
    # from tsfresh import select_features
    # from multiprocessing import freeze_support
    # # print(df.dtypes)
    # freeze_support()
    # # print(df.index.tolist())
    # # features = extract_relevant_features(x, y, column_id="NUM", column_sort="DATE_TIME")
    # features = extract_features(x, column_id="NUM", column_sort="DATE_TIME")
    # # features = extract_features(x, column_id="DATE_TIME", column_sort="DATE_TIME")
    # print(f"FEATURES.info = \n{features.info}")
    # print(f"FEATURES.shape = \n{features.shape}")
    # print(f"FEATURES.columns = \n{features.columns}")
    # print(f"FEATURES = \n{features}")
    # #
    # impute(features)
    # filtered_featrues = select_features(features, y)
    # print(f"FILTERED_FEATURES = \n{filtered_featrues}")
    #

    from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures

    download_robot_execution_failures()
    x, y = data = tsfresh.examples.load_robot_execution_failures()
    print(x,'\n', y)

    from tsfresh import extract_features
    features = extract_features(x, column_id='id', column_sort='time')
    print(features)

    from tsfresh.utilities.dataframe_functions import impute
    impute(features)

    from tsfresh import select_features
    filtered_features = select_features(features, y)
    print(filtered_features)

    from tsfresh import extract_relevant_features
    r_features = extract_relevant_features(x, y, column_id='id', column_sort='time')
    print(r_features)
    """
