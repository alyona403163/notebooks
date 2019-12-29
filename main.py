from catboost import CatBoostRegressor
import pandas as pd
import numpy as np


train_1 = pd.read_csv('data/train_1.csv')

X_val = train_1.loc[train_1['Week'] == 9]
y_val = np.log(X_val['Adjusted_demand']+1)
X_val.drop(['Week', 'Adjusted_demand'], axis=1, inplace=True)

X_train = train_1.loc[train_1['Week'] == 8]
y_train = np.log(X_train['Adjusted_demand']+1)
X_train.drop(['Week', 'Adjusted_demand'], axis=1, inplace=True)

cat_features = ['Sales_depot_ID', 'Sales_channel_ID', 'Route_ID', 'Client_ID', 'Product_ID']

model1 = CatBoostRegressor(task_type='GPU')
model1.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, use_best_model=True)
model1.save_model('models/cb_regressor_counters_1.cbm')