import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from linear_regression import LinearRegression

data = pd.read_csv('../data/world-happiness-report-2017.csv')

train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

x_train = train_data[input_param_name].values
y_train = train_data[output_param_name].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

# plt.scatter(x_train, y_train, color='blue', label='Training Data')
# plt.scatter(x_test, y_test, color='red', label='Testing Data')
# plt.xlabel(input_param_name)
# plt.ylabel(output_param_name)
# plt.title('Training and Testing Data')
# plt.legend()
# plt.show()

num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train, y_train)
theta, cost_history = linear_regression.train(learning_rate, num_iterations)
print("Theta:", theta)
print("Cost history:", cost_history)