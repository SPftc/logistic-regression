# ================================
# Logistic Regression (from scratch) — Candy Dataset
# ================================

# 0) Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 1) Load / prepare data
# --------------------------------------------------
# - Binary classification with labels in {0,1}
# - X is an (m, n) feature matrix (no bias column yet)
# - y is an (m,) label vector
def load_data():
    df = pd.read_csv("creditcard.csv")
    X = df.drop(columns=["Time"]).values
    y = df["label"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    m, n = X.shape
    X = np.column_stack([np.ones(m), X])
    n1 = X.shape[1]
    return X, y, m, n1
X, y, m, n1 = load_data()


# 2) Utility functions: sigmoid, loss, gradient, prediction
# --------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_proba(X, w):
    return sigmoid(np.dot(X, w))


def binary_cross_entropy(y_true, y_prob, eps=1e-12):
    y_prob = np.clip(y_prob, eps, 1 - eps)  
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

def gradient(X, y_true, y_prob):
    m = X.shape[0]
    return (1 / m) * np.dot(X.T, (y_prob - y_true))


# 3) Initialize parameters
# --------------------------------------------------
w = np.zeros(n1)

# 4) Hyperparameters
# --------------------------------------------------
rate = 0.1
n2 = 2000
cost1 = [] 


# 5) Gradient Descent loop
# --------------------------------------------------
for i in range(n2):
    y1 = predict_proba(X, w)
    cost = binary_cross_entropy(y, y1)
    cost1.append(cost)
    grad = gradient(X, y, y1)
    w -= rate * grad


# 6) Final parameters
# --------------------------------------------------
print("params:")
print(w)


# 7) Plot: Cost vs Iterations
# --------------------------------------------------
plt.figure()
plt.plot(range(len(cost1)), cost1)
plt.xlabel("iteration")
plt.ylabel("cost")
plt.title("cost vs iterations")
plt.grid(True)
plt.show()


# 8) Plot: Cost vs 3 of the most important parameters
# --------------------------------------------------
p = np.argsort(np.abs(w[1:]))[::-1][:3] + 1
print("cost based on the parameter indices:", p)


def compute_cost_given_w(mod_w):
    return binary_cross_entropy(y, predict_proba(X, mod_w))


for i in p:
    center = w[i]
    sweep = np.linspace(center - 1.0, center + 1.0, 60)
    costs = []
    for val in sweep:
        w1 = w.copy()
        w1[i] = val
        costs.append(compute_cost_given_w(w1))

    plt.figure()
    plt.plot(sweep, costs)
    plt.xlabel(f"parameter w[{i}]")
    plt.ylabel("log loss cost")
    plt.title(f"cost for the parameters {i}")
    plt.grid(True)
    plt.show()


# 9) Inference helper
# --------------------------------------------------
def predict_label(X_new, w, threshold=0.5):
    return (predict_proba(X_new, w) >= threshold).astype(int)

p5 = predict_label(X, w)
a = (preds == y).mean()
print(f"accuracy: {a*100:.2f}%")
print(f"loss: {cost1[-1]:.4f}")
label = {0: "Not Chocolate", 1: "Chocolate"}
p10 = [label[p] for p in preds[:8500]] 
print("Sample predictions:", p10)
