import pandas as pd
from datetime import datetime

class DataLoad ():
    def __init__(self):
        # df_org : csv에서 불러온 원본 데이터셋
        self.df_org = pd.read_csv("./dataset/dataset.csv")
