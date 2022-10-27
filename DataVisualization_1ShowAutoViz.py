from autoviz.AutoViz_Class import AutoViz_Class
from DataPreprocess import DataPreprocess

class ShowAutoviz(DataPreprocess):
    def __init__(self):
        super().__init__()
        df_prcd = self.df_prcd
        print("Loading AutoViz")
        print(df_prcd)
        av = AutoViz_Class()
        av.AutoViz(filename= '',
                   dfte = df_prcd,
                   depVar = 'OK',
                   verbose = 2,
                   max_rows_analyzed= df_prcd.shape[0],
                   max_cols_analyzed= df_prcd.shape[1])

ShowAutoviz()