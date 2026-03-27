import pandas as pd 
import numpy as np
df=pd.read_csv('heart_disease_uci.csv')
print(df.head())

df = df.replace('?', np.nan)
cols_to_convert = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'thal', 'slope']

for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    num_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
    cat_cols = ['ca', 'thal', 'slope', 'fbs', 'restecg', 'exang']

for col in cat_cols:
    if df[col].mode().empty:
        print(f"{col} has no mode, filling with 0")
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
print(df.isnull().sum())

#df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
# df.drop('num', axis=1, inplace=True)
X = df.drop('target', axis=1)
y = df['target']
X = pd.get_dummies(X, drop_first=True)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42

)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

lr = LogisticRegression()
lr.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

svm = SVC(probability=True)
svm.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report
models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "Decision Tree": dt,
    "SVM": svm
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    from sklearn.metrics import roc_auc_score

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:,1]
    print(f"{name} ROC-AUC:", roc_auc_score(y_test, y_prob))

best_model = rf
print(df.columns.tolist())
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
best_model = grid.best_estimator_

from sklearn.metrics import accuracy_score, classification_report

y_pred = best_model.predict(X_test)

print("Optimized Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importances.head(10))
import pickle

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

df.to_csv("heart_disease_uci.csv", index=False)