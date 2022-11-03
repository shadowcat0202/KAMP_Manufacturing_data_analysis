from autoviz.AutoViz_Class import AutoViz_Class
from DataPreprocess import DataPreprocess

class ShowAutoviz(DataPreprocess):
    def __init__(self):
        super().__init__()


    def RUN_AUTOVIZ(self):
        # df_prcd = self.df_prcd
        df = self.df_prcd
        df.drop(columns = ['DATE_TIME', 'DATE'], inplace=True)\
        # df['OUTLIERS_MM'] =
        print(df.columns, df.dtypes, df)
        print("Loading AutoViz")
        av = AutoViz_Class()
        av.AutoViz(filename= '',
                   dfte = df,
                   depVar = 'OK',
                   verbose = 2,
                   max_rows_analyzed= df.shape[0],
                   max_cols_analyzed= df.shape[1])

ShowAutoviz().RUN_AUTOVIZ()

