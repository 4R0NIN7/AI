from utils.utils import createRandomForestModel, get_csv_data
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = get_csv_data()

milk_model, predictions, train_X, val_X, train_y, val_y  = createRandomForestModel(data, 2, 3)

print(mean_absolute_error(val_y, predictions))
