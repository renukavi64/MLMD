import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linea_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import(
  f1_score,
  accuracy_score,
  classification_report,
  confusion_matrix
)
data=load_breast_cancer()
X=pd.DataFrame(
  data.data,
  columns=data.feature_names
)
Y=pd.Series(data.target,name="target")
print("Dataset shape:",X.shape)
print("\nFirst 5 rows:")
print(X.head())
print("\nTarget values:")
print(Y.value_counts())
X_train,X_test,Y_train,Y_test=train_test_split(
  X,
  Y,
  test_size=0.20,
  random_state=42,
  stratify=Y
)
print("\nTraining samples:",X_train.shape[0])
print("Testing samples:",X_test.shape[0])
models={
  "Logostic Regression":LogisticRegression(
    max_iter=5000
  ),
  "KNN":KNeighborsClassifier(),
  "Decision Tree":DecisionTreeClassifier(
    random_state=42
  ),
  "SVM":SVC()
}
param_grids={
  "Logistic Regression":{
    "model_ _C":[0.1,1,10],
    "model_ _solver":["liblinear"]
  },
  "KNN":{
    "model_ _n_neighbors":[3,5,7,9]'
    "model_ _weights":["uniform","distance"]
  },
  "Decision Tree":{
    "model_ _max_depth":[None,3,5,10],
    "model_ _min_samples_split":[2,5,10]},
  "Random Forest":{
    "model_ _n_estimators":[100,200],
    "model_ _max_depth":[None,5,10],
    "model_ _min_samples_split":[2,5]
  },
  "SVM":{
    "model_ _C":[0.1,1,10],
    "mosdel_ _kernel":["linear","rbf"],
    "model_ _gamma":["scale","auto"]
  }
}
results=[]
best_models={}
for model_name,model in models.items():
  print("\n" + "=" * 60)
