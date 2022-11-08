from DataPreprocess import DataPreprocess

import matplotlib.pyplot as plt

import seaborn as sns
from datetime import datetime

class GlanceViz(DataPreprocess):
    def __init__(self):
        super().__init__()

    def PLOT_DFPRCD_OKPROB_BYTIMEFORMATS(self):
        """
        전처리된 데이터프레임을 기반으로 기간/시간별 OK 비율을 보여주는 그래프
        :return: NONE
        """
        ls_dateCols = self.ls_newDtCols
        df = self.df_prcd

        f, ax = plt.subplots(1, len(ls_dateCols), figsize=(20, 10))
        for i, col_date in enumerate(ls_dateCols):
            df_part = df[[col_date, self.name_target]]
            grp = df_part.groupby([col_date], as_index=False).mean()

            sns.barplot(x=grp[col_date], y=grp[self.name_target], ax=ax[i])
            # sns.lineplot(x=grp[col_date], y=grp[self.name_target], ax=ax[i])
        f.tight_layout()
        f.suptitle(f"{self.name_target} PROBABILITY BY TIME")
        plt.show()

    def plot_NG_byTwoFeatures(self, colName_feat1, colName_feat2, period = 'ALL', drawLine_at= ('ax2', 0), y_min=None, load_dfORG = False):
        """
        주어진 두 개의 피처에 따라 데이터를 시각화
        :param colName_feat1: (STR ONLY)특징 1 칼럼이름
        :param colName_feat2: (STR ONLY) 특징 2 칼럼이름
        :param period: ALL or 날짜("mm/dd"). All은 전체기간 조회, 날짜입력 시 주어진 날짜의 데이터만 조회
        :param drawLine_at: (TUPLE). (ax1 or ax2, 본인이 가로로 선을 긋고싶은 y값). 희망하지 않을 경우 None
        :param y_min: (TUPLE). (ax1 y 최소값, ax2 y 최소값). None 입력시 default 로 나옴
        :param load_dfORG: (BOOL). False일 경우 전처리 된 데이터를 불러옴. True일 경우 전처리 전 데이터를 불러옴
        :return: None
        """
        # 데이터 프레임 결정: 기본값은 전처리. load_dfORG == True일 경우 원본데이터를 불러옴
        if load_dfORG is False:
            df = self.df_prcd
        else:
            df = self.df_org

        # 불러 오려고하는 기간. ALL일 경우 전 기간. 특정 날짜로 조회하려면 'mm/dd' 형식의 String 입력
        if period == 'ALL': # ALL 인 경우
            df = df

        else: # 특정날짜로 조회하려는 경우
            data_date = period.split('-') # 스플릿
            user_dates = []
            for date in data_date:
                extracted = date.split('/') # 스플릿
                month, day = map(int, extracted) # STR >> INT
                user_date = datetime(2020, month, day).date() # Date 타입으로 변경
                user_dates.append(user_date)

            date_from, date_to = min(user_dates), max(user_dates)

            df = df[(df['DATE'] >= date_from) & (df['DATE'] <= date_to)]  # 지정 날짜의 데이터 불러오기

        # 축 2개 생성
        ax1, ax2 = plt.gca(), plt.gca().twinx()

        """
        STYLE 을 이용해서 시각화할 경우
        """
        import matplotlib.style as style
        style.use('seaborn-dark-palette')
        # style.use('seaborn-colorblind')
        # NG는 Scatter로 표현
        ax1.scatter(x='DATE_TIME', y=colName_feat1, data= df[df['OK']==0], color = 'red', label = "NG", alpha=0.9, marker='X')
        # Feature 1 Plot
        ax1.plot(df['DATE_TIME'], df[colName_feat1], label = colName_feat1, alpha=0.2)
        # Feature 2 Plot
        ax2.plot(df['DATE_TIME'], df[colName_feat2], label=colName_feat2, alpha=0.7)

        """
        컬러 직접 지정할 경우

        # NG는 Scatter로 표현
        ax1.scatter(x='DATE_TIME', y=colName_feat1, data= df[df['OK']==0], color='red', label = "NG", alpha=0.9, marker='X')

        # Feature 1 Plot
        ax1.plot(df['DATE_TIME'], df[colName_feat1], color='black', label = colName_feat1, alpha=0.9)

        # Feature 2 Plot
        ax2.plot(df['DATE_TIME'], df[colName_feat2], color='y', label=colName_feat2, alpha=0.2)
        """


        if drawLine_at[0] == 'ax1':
            ax1.plot(df['DATE_TIME'], [drawLine_at[1]]*len(df['DATE_TIME']), color='green', label = f"Your Line({drawLine_at[1]})", linestyle = "--", alpha=1)

        elif drawLine_at[1] == 'ax2':
            ax2.plot(df['DATE_TIME'], [drawLine_at[1]]*len(df['DATE_TIME']), color='green', label = f"Your Line({drawLine_at[1]})", linestyle = "--", alpha=1)
        elif drawLine_at == (None, None):
            pass
        else:
            raise KeyError


        # 각 축의 y 최소값 설정
        if y_min[0] != None:
            ax1.set_ylim(y_min[0])

        if y_min[1] != None:
            ax2.set_ylim(y_min[1])

        # 기타 설정 - 아래는 설명 생략하겠음
        plt.title(f"{colName_feat1} - {colName_feat2}")
        plt.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax1.set_ylabel(colName_feat1)
        ax2.set_ylabel(colName_feat2)
        plt.show()


    def boxplot_feature_byTimeFormats(self, colName_feat, load_dfORG = False):
        """
        주어진 특징의 값을 박스플랏화
        :param colName_feat: (STR ONLY) 특징의 칼럼이름
        :param load_dfORG: (BOOL) 전처리된 데이터 (False), 원본 데이터 (True)
        :return: None
        """
        ls_dateCols = self.ls_newDtCols

        if load_dfORG is False:
            df = self.df_prcd
        else:
            df = self.df_org

        f, ax = plt.subplots(1, len(ls_dateCols), figsize=(20,10))
        for j, col_dt in enumerate(ls_dateCols):
            sns.boxplot(x=df[col_dt], y=df[colName_feat], ax=ax[j])
        f.tight_layout()
        f.suptitle(f'{colName_feat} by date type data')
        plt.show()

    def plot_dfprcd_distribution(self, df, colName):
        """
        분포 그래프
        :param df:
        :param colName:
        :return:
        """
        sns.kdeplot(df[colName])
        plt.show()


gv = GlanceViz()

# 특징별 분포 BOX PLOT
# gv.boxplot_feature_byTimeFormats(colName_feat='MELT_WEIGHT', load_dfORG=False)

# 시간포맷에 따른 OK 확률 시각화
# gv.PLOT_DFPRCD_OKPROB_BYTIMEFORMATS()


# 전체 기간 NG 조회
# gv.plot_NG_byTwoFeatures(colName_feat1='MELT_WEIGHT', colName_feat2='MELT_TEMP',period='ALL', drawLine_at=(None, None), y_min=(None, None), load_dfORG=False)

# 특정 날짜 NG 조회
gv.plot_NG_byTwoFeatures(colName_feat1='MELT_TEMP', colName_feat2='MELT_WEIGHT',period='3/22-3/23', drawLine_at=('ax1', 50), y_min=(None, None), load_dfORG=False)

