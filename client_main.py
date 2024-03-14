import random
import numpy as np
import client
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets as data
import warnings

warnings.filterwarnings("ignore")

'''
# SHAREEDB Dataset 
# read data from dataset
main_data_frame = pd.read_csv("ExtractedFeatures.csv")

# Splitting Data Between Features & Results
x = main_data_frame.iloc[:, 2:].values
y = main_data_frame.iloc[:, 1].values

# Transform the dataset using SMOTE
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)

# randomize selection of the data
# this snippet is very specific to this case since i am using
# the same dataset for all clients

# Determine the number of items to select
n = int(len(x) // 3)
# Get the indices of n random items
indices = random.sample(range(len(x)), n)

# Select the corresponding items from x and y
x = x[indices]
y = y[indices]

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Feature Scaling
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
'''

'''
# read the simulated dataset
main_data_frame = pd.read_csv("simulated_dataset.csv")

# Splitting Data Between Features & Results
x = main_data_frame.iloc[:, :-1].values
y = main_data_frame.iloc[:, -1].values

# select only third of the data instances
n = int(len(x) // 3)

# Get the indices of n random items
indices = random.sample(range(len(x)), n)

# Select the corresponding items from x and y
x = x[indices]
y = y[indices]

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
'''


# Surgical Deepnet
main_data_frame = pd.read_csv("Surgical-deepnet.csv")
# Splitting Data Between Features & Results
x = main_data_frame.iloc[:, :-1].values
y = main_data_frame.iloc[:, -1].values

# Transform the dataset using SMOTE
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)

# select only third of the data instances
n = int(len(x) // 25)
# Get the indices of n random items
indices = random.sample(range(len(x)), n)
# Select the corresponding items from x and y
x = x[indices]
y = y[indices]

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Feature Scaling
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)


# kernel can be "linear" or "hingeloss"
client = client.Client(host="localhost", port=61297, buffer_size=4096,
                       receive_timeout=1000, print_incoming_messages=False, print_sent_messages=False,
                       x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
client.create_socket()
if client.connect():
    client.join_cycle()
