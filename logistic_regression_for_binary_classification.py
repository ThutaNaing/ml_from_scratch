from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import csv


class LogisticRegression:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.theta = np.array([])
        self.cost_history = []

    def get_m(self):
        return len(self.y)

    def add_intercept_to_X(self):
        intercept = np.ones((self.X.shape[0], 1))
        return np.hstack((intercept, self.X))

    def calculate_z(self):
        return np.dot(self.X, self.theta)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cost_function(self):
        m = self.get_m()
        y_predict = self.sigmoid(self.calculate_z())
        y_predict = np.clip(y_predict, 1e-15, 1 - 1e-15)
        return -(1/m) * np.sum(self.y * np.log(y_predict) + (1-self.y) * np.log(1 - y_predict))

    def gradient_descent(self):
        m = self.get_m()
        return (1/m) * np.dot(self.X.T, self.sigmoid(self.calculate_z()) - self.y)

    def train(self, learning_rate=0.0001, iterations=1000000, tolerance=1e-9):
        self.cost_history = []
        self.theta = np.zeros(self.X.shape[1])
        prev_cost = float('inf')
        for i in range(iterations):
            gradient = self.gradient_descent()
            self.theta -= learning_rate * gradient

            #compute cost
            cost = self.cost_function()
            self.cost_history.append(cost)
            if abs(prev_cost - cost) < tolerance:
                print(f"Converged at iteration {i}.")
                break
            prev_cost = cost

    def predict(self, test_X, threshold=0.5):
        intercept = np.ones((test_X.shape[0], 1))
        bais_added_X = np.hstack((intercept, test_X))
        z = bais_added_X @ self.theta
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid >= threshold


# Generate data
# X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_clusters_per_class=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# lg = LogisticRegression(X_train, y_train)
# lg.X = lg.add_intercept_to_X()
# lg.train()
#
# predictions = lg.predict(X_test)
# print(f"Prediction: {predictions}")
# pred_as_int = [1 if pred else 0 for pred in predictions]
# print(f"pred_as_int: {np.array(pred_as_int)}")
# print(f"Actual y: {y_test}")
# # Accuracy
# accuracy = np.mean(predictions == y_test)
# print(f"Accuracy: {accuracy:.2f}")
#
# plt.figure(figsize=(8, 5))
# plt.plot(lg.cost_history, color='blue', linewidth=2)
# plt.xlabel("Iterations", fontsize=12)
# plt.ylabel("Cost (Log Loss)", fontsize=12)
# plt.title("Cost vs. Iterations (Well-Tuned Model)", fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()


class DataFetcherFromFiles:

    def read_csv_for_data_train(self, file_name):
        with open(file_name, 'r') as file:
            x, y = [], []
            reader = csv.reader(file)
            for row in reader:
                x_vec = []
                for i, cell in enumerate(row):
                    if i == len(row) - 1:
                        # y.append(int(float(row[len(row) - 1])))
                        y.append(float(row[len(row) - 1]))
                    else:
                        x_vec.append(float(row[i]))
                x.append(x_vec)

            X_train = np.array(x)
            y_train = np.array(y)
            print(X_train.shape)
            print(y_train.shape)
            return X_train, y_train

    def test_the_algoritham(self):
        X_train, y_train = self.read_csv_for_data_train('../../kaggle_data_sets/'
                                                        'diabetics_prediction_logistic_regression_datasets/train.csv')
        X_test, y_test = self.read_csv_for_data_train('../../kaggle_data_sets/'
                                                        'diabetics_prediction_logistic_regression_datasets/test.csv')
        lg = LogisticRegression(X_train, y_train)
        lg.X = lg.add_intercept_to_X()
        lg.train()

        predictions = lg.predict(X_test)
        print(f"Prediction: {predictions}")
        pred_as_int = [1 if pred else 0 for pred in predictions]
        print(f"pred_as_int: {np.array(pred_as_int)}")
        print(f"Actual y: {y_test}")
        # Accuracy
        accuracy = np.mean(predictions == y_test)
        print(f"Accuracy: {accuracy:.2f}")

        plt.figure(figsize=(8, 5))
        plt.plot(lg.cost_history, color='blue', linewidth=2)
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("Cost (Log Loss)", fontsize=12)
        plt.title("Cost vs. Iterations (Well-Tuned Model)", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()


df = DataFetcherFromFiles()
df.test_the_algoritham()

