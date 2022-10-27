import pandas as pd
import Errors

class DataTransformation():
    def __init__(self):
        pass

    def transfCol_num2sec(self, df, colName_old, colName_new):
        """
        주어진 데이터 프레임 내 칼럼의 값에 6을 곱한 뒤 60을 나눈 나머지로 값을 변환. 새로운 이름을 적을경우 기존의 칼럼은 삭제됨.
        :param df: 값을 변경할 칼럼이 포함된 데이터프레임
        :param colName_old: 값을 변경할 칼럼의 이름 (str)
        :param colName_new: 값을 변경한 뒤 칼럼의 이름 (str)
        :return: 해당 칼럼의 값이 변경된 뒤의 데이터 프레임
        """
        """
        * 이유/근거 - 데이터셋 가이드북 p.8
        "수집주기 : 사이클타임 약 6초" >> row(NUM) 별로 6초의 간격이 있음. 따라서 시점의 초로 활용가능함 
        """

        df[colName_new] = df[colName_old] *6 %60
        df.drop(columns=[colName_old], inplace=True)

        return df

    def combineCol_datetimeNsec(self, df, colName_datetime, colName_sec, colName_new):
        """
        데이터 프레임 내 Y:M:D H:M 타입 데이터와 S데이터를 결합하여 새로운 칼럼 생성. 기존의 두 칼럼은 삭제됨
        :param df: 변경을 적용할 대상의 데이터 프레임
        :param colName_datetime: Y:M:D H:M 타입 데이터 칼럼
        :param colName_sec: S 타입 데이터 칼럼
        :param colName_new: 결합한 데이터 칼럼의 이름
        :return: 변경이 적용된 데이터 프레임
        """
        df[colName_new] = pd.to_datetime(df[colName_datetime].astype(str)+':'+df[colName_sec].astype(str))
        df[colName_new] = pd.to_datetime(df[colName_new], format = '%Y-%m-%D %I:%M:%S' )
        df.drop(columns=[colName_datetime, colName_sec], inplace=True)

        return df

    def multiplyCol_byNum(self, df, colName, by_num):
        if isinstance(colName, str):
            df[colName] = df[colName]*by_num
        elif isinstance(colName, list):
            for col in colName:
                df[col] = df[col]*by_num
        else:
            raise Errors.CheckType_StrList

        return df

    def replaceCol_withDict(self, df, colName_old, colName_new, dict):
        df[colName_new] = df[colName_old].replace(dict)
        df.drop(columns=[colName_old], inplace=True)

        return df



    # def transfCol_str2date(self, df, colName_old, colName_new):
    #     df[colName_new] =

