import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("/Users/priyanshutuli/Downloads/online_shoppers_intention.csv")
data = pd.get_dummies(df,drop_first = True)
le = LabelEncoder()
data['Revenue'] = le.fit_transform(data['Revenue'])
selected_feats = ['Administrative', 'Administrative_Duration', 'ProductRelated', 'ExitRates', 'PageValues']
X = data[selected_feats]
y = data['Revenue']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
model_rf = RandomForestClassifier(criterion= 'entropy',n_estimators = 1000, max_depth = 4)
model_rf.fit(X_train,y_train)
pickle.dump(model_rf,open('final_model.pkl','wb'))