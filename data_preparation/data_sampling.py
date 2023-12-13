# Sampling 1% of total users for efficiently developing this project. Not necessary to run this script again.
import pandas as pd
import numpy as np


df = pd.read_csv('../data/UserBehavior.csv')
df.columns = ['UserID', 'ItemID', 'ItemCategoryID', 'BehaviorType', 'TimeStamp']

# Sample 1% users
np.random.seed(551)
sampled_UserID = np.random.choice(df['UserID'].unique(), size=int(0.01*len(df['UserID'].unique())))

# Sample dataframe of sampled users
sampled_df = df[df['UserID'].isin(sampled_UserID)]

sampled_df.to_csv('../data/Sampled_UserBehavior.csv')
