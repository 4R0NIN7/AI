import pandas as pd
import tensorflow as tf
# Convention
# X - features  - the columns that are inputted into our model (and later used to make predictions)
# Y - prediction - the column we want to predict

# describe - describes the data set
# head - shows the top few rows.

melbourne_file_path = 'intro_to_machine_learing/data/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data = melbourne_data.dropna(axis=0)

Y = melbourne_data.Price 
features =  ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'] 
X =  melbourne_data[features] 

# print(X.describe())
# print(X.head())


# overfitting - a model matches the training data almost perfectly, but does poorly in validation and other new data.
# When we divide the data amongst many leaves, we also have fewer data in each leaf.\
# Leaves with very few houses will make predictions that are quite close to those data's actual values, 
# but they may make very unreliable predictions for new data (because each prediction is based on only a few data).

# underfitting - When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data
# At an extreme, if a tree divides data into only 2 or 4, each group still has a wide variety of data. 
# Resulting predictions may be far off for most data, even in the training data (and it will be bad in validation too for the same reason)