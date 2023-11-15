import pandas as pd

df = pd.read_csv('../data/UserBehavior10000.csv').iloc[:,1:]
df.columns = ['UserID', 'ItemID', 'ItemCategoryID', 'BehaviorType', 'TimeStamp']
print(df)


