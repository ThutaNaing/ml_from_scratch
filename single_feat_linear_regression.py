import matplotlib.pyplot as plt
import csv
import numpy as np


def test_data_to_plot(x, y, w_opt, b_opt):
    # Print results
    print(f"Slope (w): {w_opt}")
    print(f"Intercept (b): {b_opt}")

    # Create the scatter plot
    plt.scatter(x, y, label='Data Points')
    plt.plot(x, x * w_opt + b_opt, color='red', label="Regression Line")
    plt.xlabel('TotRmsAbvGrd')
    plt.ylabel('SalePrice')
    plt.title('Housing Prices')
    plt.legend()
    plt.show()


class SingleFeatLinearRegression:

    def __init__(self):
        self.rooms = []
        self.prices = []
        self.y = []
        self.x = []
        self.w = 0
        self.b = 0

    def read_csv_for_data_train(self, file_name):
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.x.append(int(row[0]))
                self.y.append(int(row[1]))

            self.x = np.array(self.x)
            self.y = np.array(self.y)
            print(self.x)
            print(self.y)

    def predict(self):
        return self.x * self.w + self.b

    # Calculate cost (Mean Squared Error)
    def cal_cost_function(self):
        m = len(self.x)
        y_pred = self.predict()
        return (1/(2*m)) * np.sum((y_pred - self.y) ** 2)

    # Calculate Gradient Descent for w and b
    def derivative_of_parameters(self):
        m = len(self.x)
        dw = (1/m) * np.sum((self.predict() - self.y) * self.x)
        db = (1/m) * np.sum(self.predict() - self.y)
        return dw, db

    def train(self, learning_rate=0.001, epochs=10000, tolerance=0.0001, lambda_=0.1):
        m = len(self.x)
        cost_history = []
        for epoch in range(epochs):
            # Calculate cost (Mean Squared Error with L2 regularization)
            # cost = self.cal_cost_function() + (lambda_ / (2 * m)) * (self.w ** 2)
            cost = self.cal_cost_function()
            cost_history.append(cost)

            dw, db = self.derivative_of_parameters()
            # dw = dw + (lambda_ / m) * self.w
            self.w = self.w - (learning_rate * dw)
            self.b = self.b - (learning_rate * db)

            # Check for convergence (basic)
            if epoch > 0 and abs(cost_history[epoch] - cost_history[epoch - 1]) < tolerance:
                break
        return self.w, self.b, cost_history

    def process_test_data(self, file_name):
        with open(file_name, 'r') as file:
            reader = csv.reader(file)
            self.x = []
            for row in reader:
                self.x.append(int(row[0]))
            self.x = np.array(self.x)
            y_pred = self.predict()
            # Plot the results
            plt.scatter(self.x, self.y, label="Data")
            plt.plot(self.x, y_pred, color='red', label="Regression Line")
            plt.xlabel("Number of Rooms")
            plt.ylabel("Price")
            plt.legend()
            plt.show()


single_feat_ln_reg_model = SingleFeatLinearRegression()
single_feat_ln_reg_model.read_csv_for_data_train('../../kaggle_data_sets/home-data-for-ml-course'
                                                 '/purified_two_feat_train.csv')
w_opt, b_opt, cost_history = single_feat_ln_reg_model.train()
print("Optimized weight : ", w_opt)
print("Optimized bias : ", b_opt)
test_data_to_plot(single_feat_ln_reg_model.x, single_feat_ln_reg_model.y, w_opt, b_opt)
single_feat_ln_reg_model.process_test_data('../../kaggle_data_sets/home-data-for-ml-course/purified_two_feat_test.csv')


