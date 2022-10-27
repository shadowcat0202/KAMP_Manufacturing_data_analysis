import pandas as pd

from DataPreProcess_1DataLoad import DataLoad
from DataPreprocess_2DataCleaning import DataCleaning
from DataPreprocessing_5DataTransformation import DataTransformation


dl = DataLoad() # 데이터셋 불러오는 클래스
dc = DataCleaning() # 데이터 정제하는 클래스
dt = DataTransformation() # 데이터 변환하는 클래스

updated = f"10.26.22"
note = """ 
** 업데이트 내역/예정**
(예정)
- 이상치(outliers) 제거

(10.26.22)
- NUM 값을 초로 변경 (10.26.22)
- 온도, 회전속도에 0.1 곱함 (10.26.22)
- TAG 칼럼명을 OK로 변경 후 값 변경 {'OK':1, 'NG':0} (10.26.22)
* END *
"""
ver_msg = f"[전처리데이터] LAST UPDATED ON {updated}\n{note}"

class DataPreprocess():
    def __init__(self):
        # 원본으로 불러온 데이터
        self.df_org = dl.df_org

        # 최종결과물: 전처리 된 데이터 프레임 (_prcd == processed == 처리된)
        self.df_prcd = self.processed_dataframe()
        print(ver_msg)


    def processed_dataframe(self):
        df_prcd = self.df_org.copy()

        # NUM을 초(second)로 변환
        df_prcd['SEC'] = df_prcd['NUM']*6 % 60
        """
        * 이유/근거 - 데이터셋 가이드북 p.8
        "수집주기 : 사이클타임 약 6초" >> row(NUM) 별로 6초의 간격이 있음. 따라서 시점의 초로 활용가능함 
        
        """

        df_prcd['DATE_TIME'] = pd.to_datetime(df_prcd['STD_DT'].astype(str)+':'+df_prcd['SEC'].astype(str))
        df_prcd['DATE_TIME'] = pd.to_datetime(df_prcd['DATE_TIME'], format = '%Y-%m-%D %I:%M:%S' )
        df_prcd.drop(columns=['STD_DT', 'SEC' ,'NUM'], inplace=True)

        # 온도, 속도에 0.1을 곱해줌
        df_prcd['MELT_TEMP'] = df_prcd['MELT_TEMP']*0.1
        df_prcd['MOTORSPEED'] = df_prcd['MOTORSPEED']*0.1
        """
        * 이유/근거 - 데이터셋 가이드북 p.9
        "용해온도와 교반속도 데이터는 소수점 1자리가 생략되어 있기 때문에 값 nnn은 실제로 nn.n을 의미한다(예: 501 → 50.1℃)"
        """

        # TAG 칼럼명을 OK?로 변경후 원핫인코딩
        df_prcd['TAG'].replace({'OK':1, 'NG':0}, inplace=True)
        df_prcd.rename(columns={'TAG':'OK'}, inplace=True)

        # 칼럼 재정렬
        df_prcd = df_prcd[['DATE_TIME', "MELT_TEMP", "MOTORSPEED", "MELT_WEIGHT", "INSP", "OK"]]

        return df_prcd

