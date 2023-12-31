# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split

data = pd.read_csv("../data/Sampled_UserBehavior.csv")

df=data.iloc[:,1:5]
value_mapping = {"pv":1,"fav":2,"cart":3,"buy":4}
df_no_duplicates = df.drop_duplicates()

df_no_duplicates["BehaviorType"]=df_no_duplicates["BehaviorType"].map(value_mapping)
result = df_no_duplicates.groupby(['UserID', 'ItemID']).agg({'BehaviorType': 'sum'}).reset_index()

from scipy.sparse import coo_matrix

userID = result["UserID"].values
itemID = result["ItemID"].values
interactions = np.ones(userID.shape[0])
unique_user_ids = np.unique(userID)
unique_item_ids = np.unique(itemID)

user_id_to_index = {user_id: index for index, user_id in enumerate(unique_user_ids)}
item_id_to_index = {item_id: index for index, item_id in enumerate(unique_item_ids)}

user_indices = [user_id_to_index[user_id] for user_id in userID]
item_indices = [item_id_to_index[item_id] for item_id in itemID]

interactions_matrix = coo_matrix((interactions, (user_indices, item_indices)), shape=(len(unique_user_ids), len(unique_item_ids)))
train, test = random_train_test_split(interactions_matrix, random_state=123123)

model = LightFM(loss='bpr')
model.fit(train, epochs=30, num_threads=2)

from lightfm.evaluation import precision_at_k, auc_score
precision = precision_at_k(model, test, k=5).mean()
auc = auc_score(model, test).mean()

