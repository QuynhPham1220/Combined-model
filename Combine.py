import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer


data = 'Training_data.xlsx'
df = pd.read_excel(data)
df.info()
df.head()

y_train = df['Classification']
X_train = df.drop(['Epitope','HLA','HLA','Classification','Group'], axis=1)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay
accuracy10x = cross_val_score(model, X_train, y_train, scoring='roc_auc' ,cv=10)
[print(x) for x in accuracy10x]
accuracy10x.mean()

sensitivity = cross_val_score(model, X_train, y_train, scoring='recall' ,cv=10)
[print(x) for x in sensitivity]
sensitivity.mean()

specificity = make_scorer(recall_score, pos_label=0)
specificity10x = cross_val_score(model, X_train, y_train, cv=10, scoring=specificity)
[print(x) for x in specificity10x]
specificity10x.mean()

proba = cross_val_predict(model, X_train, y_train, cv=10, method='predict_proba')
[print(x) for x in proba[:, 1]] 

data = 'Testing_data.xlsx'
df_test = pd.read_excel(data)
df_test.info()
df_test.head()

y_test = df_test['Classification']
x_test = df_test.drop(['Epitope','HLA','CDR3','Classification','Group'], axis=1)

y_pred_test = model.predict(x_test)
y_pred_test_proba = model.predict_proba(x_test)
[print(x) for x in y_pred_test_proba[:,1]]

