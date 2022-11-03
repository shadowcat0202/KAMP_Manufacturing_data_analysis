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
