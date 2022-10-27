# pip install pyod

from DataPreProcess_1DataLoad import DataLoad

"""
데이터 클리닝
- 대상: noise, artifact, precision, bias, outlier, missing values, inconsistent value, duplicate data
- 진단
    - missing values 존재하지 않음
    - duplicate data 존재하지 않음
    
"""

class DataCleaning(DataLoad):
    def __init__(self):
        super().__init__()

    def no_missingValues(self):
        """
        원본 데이터베이스에 null 값이 있는지 확인.
        :return: True (null이 데이터프레임 내 하나도 없을 경우) or False (else)
        """
        result = True if self.df_org.isnull().sum() ==0 else False
        return result

    def no_duplicates(self):
        """
        원본 데이터베이스에 중복된 데이터가 있는지 확인.
        :return: True (null이 데이터프레임 내 하나도 없을 경우) or False (else)
        """
        result = True if self.df_org.duplicated().sum() == 0 else False
        return result

