import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import logistic
from data_preparation import lr_data_prep
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

np.random.seed(42)
# Read in and prepare the data
df = pd.read_csv('../data/UserBehavior-Without-Timestamp.csv').iloc[:, 1:]
df = df[df['ItemCategoryID'] == 4921805]
df = lr_data_prep(df)

# Split into training and test data
# X = df[['pv', 'cart', 'fav', 'pv_cart', 'pv_fav', 'cart_fav', 'pv_cart_fav']]
X = df[['pv', 'cart', 'fav']]
y = df['buy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=551)


# Define the logistic function
def logistic_function(x):
    return 1 / (1 + np.exp(-x))


# Define the likelihood function for logistic regression
def likelihood(theta, X, y):
    logits = np.dot(X, theta)
    return np.prod(logistic.pdf(y * logits))


# Define the prior distribution for logistic regression parameters
def prior(theta):
    # prior_mean = np.array([9, -13, -12, -13])
    prior_mean = np.array([10, -10, -10, -10])
    # prior_mean = np.array([9,-13,-12,-13,14,15,13,-14])
    return np.prod(np.exp(-(theta-prior_mean)**2 / 2) / np.sqrt(2 * np.pi))


# Define the posterior distribution (proportional to likelihood times prior)
def posterior(theta, X, y):
    return likelihood(theta, X, y) * prior(theta)


# Metropolis-Hastings algorithm for Bayesian logistic regression
def metropolis_hastings_logistic(iterations, X, y, proposal_sd):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)  # Initial parameters (including intercept)

    samples = [theta.copy()]

    for _ in range(iterations):
        current_theta = samples[-1]

        # Propose a new sample from a multivariate normal distribution
        proposed_theta = np.random.multivariate_normal(mean=current_theta, cov=np.eye(n_features) * proposal_sd)

        # Calculate acceptance ratio
        acceptance_ratio = min(1, posterior(proposed_theta, X, y) / posterior(current_theta, X, y))

        # Accept or reject the proposed sample
        if np.random.rand() < acceptance_ratio:
            samples.append(proposed_theta)
        else:
            samples.append(current_theta)

    return np.array(samples)


def evaluate_lr(X, y, params):
    y_pred = 1/(1+np.exp(-X @ np.mean(params, axis=0)))
    y_pred = [1 if y_pred > 0.5 else 0 for y_pred in y_pred]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    return accuracy, precision, recall


if __name__ == '__main__':
    # Number of iterations and proposal standard deviation
    iterations = 20000
    proposal_sd = 0.1

    # Add intercept term to the feature matrix
    X_train_with_intercept = np.column_stack((np.ones(X_train.shape[0]), X_train))
    X_test_with_intercept = np.column_stack((np.ones(X_test.shape[0]), X_test))

    # Run Metropolis-Hastings algorithm for logistic regression
    samples = metropolis_hastings_logistic(iterations, X_train_with_intercept, y_train, proposal_sd)

    for i in range(X.shape[1]+1):
        print(np.mean(samples[:, i]))
        print(np.percentile(samples[:, i], [2.5, 97.5]))


    # Plot the trace of the parameters
    plt.figure(figsize=(10, 6))
    plt.subplot(221)
    plt.plot(samples[:, 0], label='Intercept')
    plt.subplot(222)
    plt.plot(samples[:, 1], label='Coefficient for Feature 1')
    plt.subplot(223)
    plt.plot(samples[:, 2], label='Coefficient for Feature 2')
    plt.subplot(224)
    plt.plot(samples[:, 3], label='Coefficient for Feature 3')

    # plt.title('Trace of Logistic Regression Parameters')
    # plt.xlabel('Iteration')
    # plt.ylabel('Parameter Value')
    # plt.legend()
    plt.show()

    train_accuracy, train_precision, train_recall = evaluate_lr(X_train_with_intercept, y_train, samples)
    print("Train Accuracy:", train_accuracy)
    print("Train Precision:", train_precision)
    print("Train Recall:", train_recall)


    test_accuracy, test_precision, test_recall = evaluate_lr(X_test_with_intercept, y_test, samples)
    print("Test Accuracy:", test_accuracy)
    print("Test Precision:", test_precision)
    print("Test Recall:", test_recall)