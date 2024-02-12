import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def get_csv_data():
    csv_files_path = '*.csv'
    csv_files = glob.glob(csv_files_path)
    dfs = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    return final_df

def createRandomForestModel(data: pd.DataFrame, random_state_model=2, random_state_split=1):
    y = data.realVolumeValue # what are we want to predict
    features = ['milkVolume', 'confidence', 'totalMilkVolume', 'totalRange', 'bottleSize', 'bottleState', 'tiltOutOfRange', 'motionOutOfRange']
    X = data[features] # from which columns we want to predict 
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=random_state_split) # splitting data to train & values
    milk_model = RandomForestRegressor(random_state=random_state_model) # creating model
    milk_model.fit(train_X, train_y) # training model
    predictions = milk_model.predict(val_X) # testing
    return milk_model, predictions, train_X, val_X, train_y, val_y
