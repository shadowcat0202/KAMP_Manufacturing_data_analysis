import matplotlib.pyplot as plt
import seaborn as sns

from DataPreprocess import DataPreprocess

class VisualizedGlance(DataPreprocess):
    def __init__(self):
        super().__init__()


    # def showAll_inLineChart(self):
    #     df = self.df_prcd
    #     # f, ax = plt.subplots(1, 1, figsize = (20, 8))
    #     plt.plot(df['MELT_TEMP'])
    #     plt.show()

df = DataPreprocess().df_prcd
df_nonStop = df.loc[df['MOTORSPEED'] > 0]
# df_stop = df.loc[df['MOTORSPEED'] == 0]
plt.plot(df_nonStop['MOTORSPEED'])
# plt.plot(df_stop['MOTORSPEED'])

# f, ax = plt.subplots(1, 1, figsize = (20, 8))
# plt.plot(df['MELT_TEMP'], color = 'g')
# plt.plot(df['MOTORSPEED'], color = 'r')
# plt.plot(df['MELT_WEIGHT'], color = 'c')
# plt.plot(df['INSP'], color = 'b')
plt.show()

