import numpy as np
import joblib
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


def relu(x):
    return np.maximum(0, x)


def forward(coefs, intercepts, x):
    a = x.copy()
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        z = a @ W + b
        a = z if i == len(coefs) - 1 else relu(z)
    return a


# Retrain with identical config so weights match weights.bin
X, y = make_classification(n_samples=10000, n_features=10, random_state=40)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    max_iter=10000,
    random_state=40,
)
mlp.fit(X_train_scaled, y_train)

print("Python raw outputs for first 5 test samples:")
print("(compare these against C output printed by ./build/micrograd)\n")

for i, x in enumerate(X_test_scaled[:5]):
    out = forward(mlp.coefs_, mlp.intercepts_, x)
    print(f"sample {i}: {out[0]:.10f}")

x = X_test_scaled[0:1]

N = 100000
start = time.perf_counter()
for _ in range(N):
    forward(mlp.coefs_, mlp.intercepts_, x)
end = time.perf_counter()

ns_per = (end - start) / N * 1e9
print(f"numpy forward pass time: {ns_per:.2f} ns per sample")

start = time.perf_counter()
for _ in range(N):
    mlp.predict(x)
end = time.perf_counter()

ns_per = (end - start) / N * 1e9
print(f"sklearn forward pass time: {ns_per:.2f} ns per sample")
