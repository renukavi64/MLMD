import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
  "Logistic Regression":LogisticRegression(
    max_iter=5000
  ),
  "KNN":KNeighborsClassifier(),
  "Decision Tree":DecisionTreeClassifier(
    random_state=42
  ),
  "Random Forest":RandomForestClassifier(
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
    "model_ _n_neighbors":[3,5,7,9],
    "model_ _weights":["uniform","distance"]
  },
  "Decision Tree":{
    "model_ _max_depth":[None,3,5,10],
    "model_ _min_samples_split":[2,5,10]
  },
  "Random Forest":{
    "model_ _n_estimators":[100,200],
    "model_ _max_depth":[None,5,10],
    "model_ _min_samples_split":[2,5]
  },
  "SVM":{
    "model_ _C":[0.1,1,10],
    "model_ _kernel":["linear","rbf"],
    "model_ _gamma":["scale","auto"]
  }
}
results=[]
best_models={}
for model_name,model in models.items():
  print("\n" + "=" * 60)
  print("Running:",model_name)
  print("="*60)
  pipeline=Pipeline([
    ("scaler",StandardScaler()),
    ("model",model)
  ])
  grid_search=GridSearchCV(
    estimator=pipeline,
    param_grid=param_grids[model_name],
    scoring="f1",
    cv=5,
  n_jobs=-1
  )
  grid_search.fit(X_train.Y_train)
  best_models[model_name]=grid_search.best_estimator_
  results.append({
    "Model":model_name,
    "Best F1 Score":grid_search.best_score_,
    "Best Parameter":grid_search.best_params_
  })
  print("Best F1 Score:",
        round(grids_search.best_score_,4))
  print("Best Parameter:",
        grid_search.best_params_)
  results_df=pd.DataFrame(results)
  results_df=results_df.sort_values(
    by="Best F1 Score",
    ascending=False
  )
  print("\n"+"="*60)
  print("MODEL COMPARISION")
  print("="*60)
  print(
    results_df[
    ["Model", "Best F1 Score"]
    ].to_string(index=False)
  )
  best_model_name=results_df.iloc[0]["Model"]
  best_model=best_models[best_model_name]
  best_cv_f1=results_df.iloc[0]["Best F1 Score"]
  print("\nBest model based on CV F1-Score:")
  print(best_model_name)
  print("\nBest CV F1-Score:")
  print(round(best_cv_f1,4))
  Y_ored=best_model.predict(X_test)
  print("\n"+"="*60)
  print("FINAL MODEL EVALUATION")
  print("="*60)
  print("Accuracy:",accuracy_score(Y_test,Y_pred))
  print("F1 Score:",f1_score(Y_test,Y_pred))
  print("\nClassification Report:")
  print(
    classification_report(
      Y_test,
      Y_pred,
      target_names=data.target_names
    )
  )
  print("\nConfusion Matrix:")
  print(confusion_matrix(Y_test,Y_pred))
  joblib.dump(best_model,"best_model.joblib")
  print("\nBest model saved successfully!")
  print("Model:",best_model_name)
  
