from DataPreprocess import DataPreprocess

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class GlanceViz(DataPreprocess):
    def __init__(self):
        super().__init__()


    def BOXPLOT_DFPRCD_VARS_BYTIME(self):
        """
        전처리된 데이터 프레임을 기반으로 기간/시간별 변수들의 형태를 시각화
        :return: NONE
        """
        ls_dateCols = self.ls_newDtCols
        ls_varCols = self.ls_vars

        df = self.df_prcd

        for i, col_var in enumerate(ls_varCols):
            f, ax = plt.subplots(1, len(ls_dateCols), figsize=(20,10))
            for j, col_dt in enumerate(ls_dateCols):
                sns.boxplot(x=df[col_dt], y=df[col_var], ax=ax[j])
            f.tight_layout()
            f.suptitle(f'{col_var} by date type data')
            plt.show()

    def PLOT_DFPRCD_OKPROB_BYTIME(self):
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
            # grp = grp
            print(grp)
            sns.barplot(x=grp[col_date], y=grp[self.name_target], ax=ax[i])
            # sns.lineplot(x=grp[col_date], y=grp[self.name_target], ax=ax[i])
        f.tight_layout()
        f.suptitle(f"{self.name_target} PROBABILITY BY TIME")
        plt.show()

    def PLOT_DFPRCD_WEIGHT_BYTIME(self):
        """
        전처리된 데이터프레임을 기반으로 기간/시간별 OK 비율을 보여주는 그래프
        :return: NONE
        """
        ls_dateCols = self.ls_newDtCols
        df = self.df_prcd

        f, ax = plt.subplots(1, len(ls_dateCols), figsize=(20, 10))
        for i, col_date in enumerate(ls_dateCols):
            df_part = df[[col_date, 'MELT_WEIGHT']]
            grp = df_part.groupby([col_date], as_index=False).min()
            # grp = df_part.groupby([col_date], as_index=False).mean()
            # grp = grp
            print(grp)
            sns.barplot(x=grp[col_date], y=grp["MELT_WEIGHT"], ax=ax[i])
            # sns.lineplot(x=grp[col_date], y=grp[self.name_target], ax=ax[i])
        f.tight_layout()
        f.suptitle(f"MELT_WEIGHT PROBABILITY BY TIME")
        plt.show()

    def plot_dfprcd_byTime(self, month, day, hour_from, hour_to):
        df = self.df_prcd
        user_date = datetime(2020, month, day).date()
        df_part = df[(df['DATE'] == user_date) & (df['HOUR'] >= hour_from) & (df['HOUR'] <= hour_to)]

        df_part_w0 = df_part[df_part['MELT_WEIGHT']==0]
        df_part_s0 = df_part[df_part['MOTORSPEED']==0]


        ax1, ax2 = plt.gca(), plt.gca().twinx()
        sns.lineplot(x=df_part['DATE_TIME'], y=df_part['MELT_WEIGHT'], ax=ax1, color='r', label = "WEIGHT", alpha=0.9)
        sns.lineplot(x=df_part_w0['DATE_TIME'], y=df_part_w0['MELT_WEIGHT'], ax=ax1, color='r', label = "WEIGHT(0)", alpha=0.9, markers='x', linewidth=0.5)
        sns.lineplot(x=df_part['DATE_TIME'], y=df_part['MOTORSPEED'], ax=ax2, color='y', label="MOTORSPEED", alpha=0.2)
        sns.lineplot(x=df_part_s0['DATE_TIME'], y=df_part_s0['MOTORSPEED'], ax=ax2, color='y', label="MOTORSPEED(0)", alpha=0.2, markers='x', linewidth=0.5)
        plt.legend(loc='right')
        plt.show()


gv = GlanceViz()
# gv.BOXPLOT_DFPRCD_VARS_BYTIME()
# gv.PLOT_DFPRCD_OKPROB_BYTIME()
# gv.PLOT_DFPRCD_WEIGHT_BYTIME()
gv.plot_dfprcd_byTime(month= 3, day= 4, hour_from=0 , hour_to= 0)