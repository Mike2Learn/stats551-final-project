import pandas as pd
import numpy as np


df = pd.read_csv('../data/UserBehavior.csv')
df.columns = ['UserID', 'ItemID', 'ItemCategoryID', 'BehaviorType', 'TimeStamp']

# Sample UserID
sampled_UserID = np.random.choice(df['UserID'].unique(), size=10000)
print(sampled_UserID)

print(df[df['UserID'] in sampled_UserID])

