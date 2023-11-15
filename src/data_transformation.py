# Run this script to generate a pivot dataframe
import pandas as pd


df = pd.read_csv('../data/Sampled_UserBehavior.csv').iloc[:, 1:]
# df = pd.read_csv('../data/UserBehavior10000.csv').iloc[:, 1:]
df.columns = ['UserID', 'ItemID', 'ItemCategoryID', 'BehaviorType', 'TimeStamp']

# Pivot by behavior type, but dropping timestamp
pivot_df = pd.pivot_table(df, values='BehaviorType', index=['UserID', 'ItemID', 'ItemCategoryID'], columns='BehaviorType', aggfunc='any')
pivot_df.reset_index(inplace=True)

pivot_df.to_csv('../data/UserBehavior-Without-Timestamp.csv')

# TODO: how to transform while keeping timestamp?

