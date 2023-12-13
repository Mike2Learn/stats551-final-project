import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from data_preparation import lr_data_prep


# Read in and preprocess the data
df = pd.read_csv('../data/UserBehavior-Without-Timestamp.csv').iloc[:, 1:]
df2 = df[df['ItemCategoryID'] == 4086613]
df = lr_data_prep(df)
df2 = lr_data_prep(df2)


# Split into training and test data
def prepare_train_test_data(df):
    X = df[['pv', 'cart', 'fav', 'pv_cart', 'pv_fav', 'cart_fav', 'pv_cart_fav']]
    # X = df[['pv', 'cart', 'fav']]
    y = df['buy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=551)

    return X_train, X_test, y_train, y_test

# Train a logistic regression model on training data
def train_lr(X_train, y_train):
    lr = LogisticRegression(random_state=551)
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)

    print("Train Accuracy:", train_accuracy)
    print("Train Precision:", train_precision)
    print("Train Recall:", train_recall)

    return lr


# Evaluate the trained model on test data. Use accuracy, precision, recall and auc as metrics.
def evaluate_lr(lr_result, X_test, y_test):

    y_test_pred = lr_model.predict(X_test)

    # Calculate evaluation metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)

    # Predict probabilities for ROC curve
    y_test_probs = lr_model.predict_proba(X_test)[:, 1]

    # Calculate AUC
    roc_auc = roc_auc_score(y_test, y_test_probs)

    # Print evaluation metrics
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", precision)
    print("Test Recall:", recall)
    print("AUC:", roc_auc)

    # Print model parameters (coefficients)
    print("Model Parameters:")
    for feature, coef in zip(X_test.columns, lr_model.coef_[0]):
        print(f"{feature}: {coef}")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_train_test_data(df2)
    lr_model = train_lr(X_train, y_train)
    evaluate_lr(lr_model, X_test, y_test)

    X_train_2, X_test_2, y_train_2, y_test_2 = prepare_train_test_data(df2)
    # evaluate_lr(lr_model, X_test_2, y_test_2)
