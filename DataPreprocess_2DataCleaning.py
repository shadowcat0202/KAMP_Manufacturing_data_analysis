# pip install pyod
# pip install kats

from DataPreProcess_1DataLoad import DataLoad

# from pyod.utils.example import visualize
# from pyod.utils.data import evaluate_print
# from pyod.models.knn import KNN

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

    # def outliers(self):
    #     clf_name = 'KNN'
    #     clf = KNN()
    #     clf.fit(self.df_org)

#
# df = DataLoad().df_org
# df = DataLoad().df_prcd

# df_part = df[['MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP', 'TAG']]
# df_part['TAG'] = df_part['TAG'].replace({'OK':1, "NG":0})
# # print(df.columns.tolist())
# print(df_part)
# clf_name = 'KNN'
# clf = KNN()
# clf.fit(df_part)
#
# df_part['Outlier'] = clf.labels_
# print(df_part.value_counts())

# visualize(clf_name, )