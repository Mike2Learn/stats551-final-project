# This script contains utils for preparing data
import pandas as pd
import matplotlib.pyplot as plt


def data_pivot():
    df = pd.read_csv('../data/Sampled_UserBehavior.csv').iloc[:, 1:]
    # df = pd.read_csv('../data/UserBehavior10000.csv').iloc[:, 1:]
    df.columns = ['UserID', 'ItemID', 'ItemCategoryID', 'BehaviorType', 'TimeStamp']

    # Pivot by behavior type, but dropping timestamp
    pivot_df = pd.pivot_table(df, values='BehaviorType', index=['UserID', 'ItemID', 'ItemCategoryID'], columns='BehaviorType', aggfunc='any')
    pivot_df.reset_index(inplace=True)

    pivot_df.to_csv('../data/UserBehavior-Without-Timestamp.csv')

# TODO: how to transform while keeping timestamp?


def lr_data_prep(input_df):
    # Mapping behaviors to binary values
    behavior_mapping = {True: 1, None: 0}
    for col in ['buy', 'cart', 'fav', 'pv']:
        input_df[col] = input_df[col].map(behavior_mapping).fillna(0)
    input_df = input_df.groupby(['UserID', 'ItemID']).sum()
    # Remove Category column
    input_df = input_df.drop('ItemCategoryID', axis=1)

    # Create cross-terms of buy, cart, and fav
    input_df['pv_cart'] = input_df['pv'] * input_df['cart']
    input_df['pv_fav'] = input_df['pv'] * input_df['fav']
    input_df['cart_fav'] = input_df['cart'] * input_df['fav']
    input_df['pv_cart_fav'] = input_df['pv'] * input_df['cart'] * input_df['fav']

    return input_df


def extract_unpopular_items(input_df):
    ItemCategory = input_df['ItemCategoryID']
    ItemCategory_counts = ItemCategory.value_counts()

    filtered_Categories = ItemCategory_counts[(ItemCategory_counts>100) & (ItemCategory_counts<300)]
    print(filtered_Categories.head(10))


if __name__ == '__main__':
    df = pd.read_csv('../data/UserBehavior-Without-Timestamp.csv').iloc[:, 1:]
    print(df.shape)
    # print(lr_data_prep(df))
    extract_unpopular_items(df)