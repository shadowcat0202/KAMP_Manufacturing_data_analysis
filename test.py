import datetime

import pandas as pd
import numpy as np

df = pd.DataFrame({'B': [np.nan, 1, 2, np.nan, 4]})
df.fillna(method='ffill', inplace=True)

if np.isnan(df.at[0, 'B']):
    df.at[0, 'B'] = "HI"
    # print(1)
# if df.loc[0, 'B'] == np.nan:
#     print('HI')

print(df)
# print(df.loc[0, 'B'])


from datetime import datetime
a = datetime(2022, 3, 22).date()
b = datetime(2022, 3, 23).date()
print(a<b)

