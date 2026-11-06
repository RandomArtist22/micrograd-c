import struct
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def relu(x):
    return np.maximum(0, x)


def forward(coefs, intercepts, x):
    a = x.copy()
    for i, (W, b) in enumerate(zip(coefs, intercepts)):
        z = a @ W + b
        if i != len(coefs) - 1:
            a = relu(z)
        else:
            a = z
    return a


# Test-data generated at random for simple classification
X, y = make_classification(n_samples=1000, n_features=10, random_state=40)

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
)

# Scaling the features for training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and Training the MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(64,),  # Using 1 layer with 64 neurons
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=40,
)
mlp.fit(X_train_scaled, y_train)

# Evaluating the model and presenting the results
y_pred = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}\n")
print("Detailed Report")
print(classification_report(y_test, y_pred))

# Extracting the trained model data
nin = mlp.n_features_in_
output_dim = mlp.n_outputs_
nlayers = len(mlp.coefs_)

# Outputting the data to cross check with the inference engine
print(f"\nnlayers: {nlayers}, nin: {nin}\n")
print(f"layer 0, neuron 0, weight 0: {mlp.coefs_[0][0, 0]}")
print(f"layer 0, neuron 0, bias: {mlp.intercepts_[0][0]}\n")

# Getting the ouput for a sample input for verification
print("\nPython raw output for a sample input:")
for x in X_test_scaled[:5]:
    output = forward(mlp.coefs_, mlp.intercepts_, x)
    print(f"Python output: {output}")

# Writing the data to the binary file
with open("weights.bin", "wb") as f:
    # Header
    f.write(struct.pack("I", nlayers))
    f.write(struct.pack("I", nin))
    for nout in mlp.hidden_layer_sizes:
        f.write(struct.pack("I", nout))
    f.write(struct.pack("I", output_dim))

    # Weights: Walking through Layers -> Neurons -> Coefficients
    for layer_weights, layer_biases in zip(mlp.coefs_, mlp.intercepts_):
        for neuron_weights, neuron_bias in zip(layer_weights.T, layer_biases):
            ws = neuron_weights.tolist()
            b = float(neuron_bias)

            f.write(struct.pack(f"{len(ws)}d", *ws))
            f.write(struct.pack("d", b))
