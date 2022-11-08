import datetime

import pandas as pd

from DataPreProcess_1DataLoad import DataLoad
from DataPreprocess_2DataCleaning import DataCleaning
from DataPreprocess_5DataTransformation import DataTransformation
from DataPreprocess_6NewFeatures import NewFeatures

from FindCycle_app_JEON import find_cycle, coundCycle

dl = DataLoad() # 데이터셋 불러오는 클래스
dc = DataCleaning() # 데이터 정제하는 클래스
dt = DataTransformation() # 데이터 변환하는 클래스
nf = NewFeatures()

class DataPreprocess():
    def __init__(self):
        # 원본으로 불러온 데이터
        self.ls_cols = dl.df_org.columns.tolist() #원본 데이터의 칼럼명 리스트
        self.ls_newDtCols = ['MONTH', 'WEEK', 'DATE', 'WEEKDAY', 'HOUR'] # 추가되는 date 데이터 칼럼 이름
        self.ls_vars = ['MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP'] # 종속변수의 리스트
        self.name_target = 'OK'
        self.df_org = dl.df_org

        # 최종결과물: 전처리 된 데이터 프레임 (_prcd == processed == 처리된)
        self.df_prcd = self.DATAFRAME_PREPROCESSED()


    def DATAFRAME_PREPROCESSED(self):
        df_prcd = dc.df_clnd.copy()


        # df_prcd = dc.df_clnd.copy()

        # NUM을 초(second)로 변환
        df_prcd = dt.transfCol_num2sec(df_prcd, colName_old='NUM', colName_new='SEC')
        """
        * 이유/근거 - 데이터셋 가이드북 p.8
        "수집주기 : 사이클타임 약 6초" >> row(NUM) 별로 6초의 간격이 있음. 따라서 시점의 초로 활용가능함 
        """

        # 날짜/시간/분 데이터(STD_DT) 와 초 데이터(SEC)를 'DATE_TIME' 칼럼으로 결합 후 불필요한 칼럼 드랍 (STD_DT, SEC, NUM)
        df_prcd = dt.combineCol_datetimeNsec(df_prcd, colName_datetime='STD_DT', colName_sec='SEC', colName_new='DATE_TIME')

        # 온도, 속도에 0.1을 곱해줌
        # df_prcd = dt.multiplyCol_byNum(df_prcd, colName = ['MELT_TEMP', 'MOTORSPEED'], by_num=0.1)
        """
        * 이유/근거 - 데이터셋 가이드북 p.9
        "용해온도와 교반속도 데이터는 소수점 1자리가 생략되어 있기 때문에 값 nnn은 실제로 nn.n을 의미한다(예: 501 → 50.1℃)"
        """

        # TAG 칼럼명을 OK?로 변경후 원핫인코딩
        df_prcd = dt.replaceCol_withDict(df_prcd, colName_old='TAG', colName_new=self.name_target, dict= {'OK':1, 'NG':0})


        # 날짜/시간 칼럼을 가지고 월/주/일/시간 등의 칼럼 추가
        df_prcd = nf.expand_date_columns(df_prcd)

        # Lag Features
        ls_vars = ['MELT_TEMP', 'MOTORSPEED', 'MELT_WEIGHT', 'INSP', 'OK'] # 종속변수의 리스트
        df_prcd, ls_lagCols = nf.generate_columns_withLagFeatures(df= df_prcd, col_feats= ls_vars, back_to= 10)

        # Window Features
        df_prcd, ls_wndwCols = nf.generate_columns_withWindowFeatures(df= df_prcd, col_feats=ls_vars, window_size= 10)

        # Cycle
        df_prcd = find_cycle(df_prcd)

        # MM 아웃라이어 하나로 통합
        ls_mmCols = [ 'OUTLIER_WTG(MM_60)', 'OUTLIER_WTG(MM_30)', 'OUTLIER_WTG(MM_15)', 'OUTLIER_WTG(MM_9)', 'OUTLIER_WTG(MM_3)']
        df_prcd['OUTLIER_WTG(MM)'] = df_prcd[ls_mmCols].sum(axis=1)
        df_prcd.drop(columns = ls_mmCols, inplace=True)

        return df_prcd

