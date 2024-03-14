import random

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import server
import warnings

warnings.filterwarnings("ignore")

host = 'localhost'
port = 61297
backlog_queue = 5
tableSize = 10  # Keys table size
minimum_clients = 3
timeout = 10
buffer_size = 4096
aggregation_method = "average_aggregation"  # "average_aggregation" # "clipped_avg_aggregator"
# "dp_avg_aggregator" "momentum_aggregation"

'''
# SHAREEDB Dataset3
# read the validation dataset to use in global model assessment
main_data_frame = pd.read_csv("ExtractedFeaturesValidation.csv", header=None)
# Splitting Data Between Features & Results
x_test = main_data_frame.iloc[:, 0:26].values
y_test = main_data_frame.iloc[:, 26:27].values
'''

'''
# read from the generated simulation
main_data_frame = pd.read_csv("simulated_dataset.csv")

# Splitting Data Between Features & Results
x = main_data_frame.iloc[:, :-1].values
y = main_data_frame.iloc[:, -1].values

# select only 1/9 of the data instances to test
n = int(len(x) // 150)

# Get the indices of n random items
indices = random.sample(range(len(x)), n)

# Select the corresponding items from x and y
x_test = x[indices]
y_test = y[indices]
'''


# Surgical Deepnet
main_data_frame = pd.read_csv("Surgical-deepnet.csv")
# Splitting Data Between Features & Results
x = main_data_frame.iloc[:, :-1].values
y = main_data_frame.iloc[:, -1].values
# select only 1/9 of the data instances to test
n = int(len(x) // 80)
# Get the indices of n random items
indices = random.sample(range(len(x)), n)
# Select the corresponding items from x and y
x_test = x[indices]
y_test = y[indices]


# define your model. Currently, only the below are supported
# ======================================================================================================================
# Model type 1 - ScikitLearn SVM
# ------------------------------------------------------------------
#model = SVC(kernel='linear', C=1.0, gamma=0.1)

# Model type 2 - ScikitLearn LogisticRegression
# ------------------------------------------------------------------
# model = LogisticRegression(penalty='none', C=3.1, solver='newton-cg', verbose=0)

# Model type 3 - ScikitLearn Gaussian Naive Bayes
# ------------------------------------------------------------------
# model = GaussianNB()  # Completed

# Model type 4 - ScikitLearn SGD Classifier
# ------------------------------------------------------------------
# model = SGDClassifier()  # Completed

# Model type 5 - ScikitLearn Neural Network (Multi Layer Perceptron)
# ------------------------------------------------------------------

model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', alpha=0.0001,
                      batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=200,
                      shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
                      momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                      beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

# ======================================================================================================================

print(type(model).__name__)
connection = server.server(host=host, port=port, table_size=tableSize, backlog_queue=backlog_queue,
                           minimum_clients=minimum_clients, timeout=timeout, buffer_size=buffer_size,
                           print_incoming_messages=False, print_sent_messages=False, print_model_summary=False,
                           print_model_performance=True, model=model, aggregation_method=aggregation_method,
                           x_test=x_test, y_test=y_test, global_rounds=3)
connection.start()