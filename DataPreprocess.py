from DataPreProcess_1DataLoad import DataLoad
from DataPreprocess_2DataCleaning import DataCleaning
from DataPreprocessing_5DataTransformation import DataTransformation


dl = DataLoad() # 데이터셋 불러오는 클래스
dc = DataCleaning() # 데이터 정제하는 클래스
dt = DataTransformation() # 데이터 변환하는 클래스


class DataPreprocess():
    def __init__(self):
        # 원본으로 불러온 데이터
        self.df_org = dl.df_org

        # 양품 데이터의 데이터 프레임
        self.df_orgOk = self.df_org.loc[self.df_org['TAG'] == 'OK']
        # 불량품 데이터의 데이터 프레임
        self.df_orgNG = self.df_org.loc[self.df_org['TAG'] == 'NG']

        # 최종결과물: 전처리 된 데이터 프레임
        self.df_prcd = self.processed_dataframe()

    def processed_dataframe(self):
        df = self.df_org
        df_prcd = df

        return df_prcd

