import numpy as np
import keras as kr
import matplotlib.pyplot as plt  # Add this import

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = kr.datasets.mnist.load_data()
print(X_train.shape)
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0
print(X_train.shape)

# One-hot encoding
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_train_onehot = one_hot(y_train, 10)
y_test_onehot = one_hot(y_test, 10)

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss function
def cross_entropy_loss(y_true, y_pred):
    assert y_pred is not None, "y_pred is None - check feed_forward()"
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m


class MultiLayerNeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # weights np arrays by layers
        self.w_hidden_input_01 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.w_hidden_input_02 = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.w_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01

        # bais np arrays
        self.bias_hidden_01 = np.random.randn(self.hidden_size)
        self.bias_hidden_02 = np.random.randn(self.hidden_size)
        self.bias_output = np.random.randn(self.output_size)

    def feed_forward(self, X):
        # First Layer Z and Activation calculation
        self.Z1_hidden = np.dot(X, self.w_hidden_input_01) + self.bias_hidden_01
        self.A1_hidden = relu(self.Z1_hidden)
        # Second Layer Z and Activation calculation
        self.Z2_hidden = np.dot(self.A1_hidden, self.w_hidden_input_02) + self.bias_hidden_02
        self.A2_hidden = relu(self.Z2_hidden)
        # Output or Third Layer Z and Activation calculation
        self.Z3_output = np.dot(self.A2_hidden, self.w_hidden_output) + self.bias_output
        self.output = softmax(self.Z3_output)
        return self.output

    def back_propagation(self, X, y):
        m = y.shape[0]
        error = self.output - y

        # Output layer gradients
        dw_output = (1 / m) * np.dot(self.A2_hidden.T, error)
        db_output = (1 / m) * np.sum(error, axis=0)

        # Second hidden layer gradients
        A2_hidden_error = np.dot(error, self.w_hidden_output.T) * relu_derivative(self.Z2_hidden)
        dw_A2_hidden = (1 / m) * np.dot(self.A1_hidden.T, A2_hidden_error)
        db_A2_hidden = (1 / m) * np.sum(A2_hidden_error, axis=0)

        # First hidden layer gradients
        A1_hidden_error = np.dot(A2_hidden_error, self.w_hidden_input_02.T) * relu_derivative(self.Z1_hidden)
        dw_A1_hidden = (1 / m) * np.dot(X.T, A1_hidden_error)
        db_A1_hidden = (1 / m) * np.sum(A1_hidden_error, axis=0)

        return dw_A1_hidden, db_A1_hidden, dw_A2_hidden, db_A2_hidden, dw_output, db_output

    def train(self, X, y, epochs, l_rate):
        for epoch in range(epochs):
            self.feed_forward(X)
            dw_A1_hidden, db_A1_hidden, dw_A2_hidden, db_A2_hidden, dw_output, db_output = self.back_propagation(X, y)

            # Update parameters
            self.w_hidden_input_01 -= l_rate * dw_A1_hidden
            self.bias_hidden_01 -= l_rate * db_A1_hidden
            self.w_hidden_input_02 -= l_rate * dw_A2_hidden
            self.bias_hidden_02 -= l_rate * db_A2_hidden
            self.w_hidden_output -= l_rate * dw_output
            self.bias_output -= l_rate * db_output

            if epoch % 100 == 0:
                loss = cross_entropy_loss(y, self.feed_forward(X))
                print(f'Epoch {epoch}, Loss: {loss}')


if __name__ == "__main__":
    nn = MultiLayerNeuralNetwork(input_size=28*28, hidden_size=128, output_size=10)
    nn.train(X_train, y_train_onehot, epochs=1000, l_rate=0.1)

    # Test the neural network
    predictions = nn.feed_forward(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    print("Predictions:", predicted_labels)
    print("Actual:", y_test)

    # Function to plot test images with predictions
    def plot_predictions(X, y_true, y_pred, num_images=10):
        plt.figure(figsize=(10, 10))
        for i in range(num_images):
            plt.subplot(5, 5, i + 1)
            plt.imshow(X[i].reshape(28, 28), cmap='gray')
            plt.title(f"True: {y_true[i]}, Pred: {y_pred[i]}")
            plt.axis('off')
        plt.show()

    # Plot some test images with their predicted and actual labels
    plot_predictions(X_test, y_test, predicted_labels, num_images=10)
