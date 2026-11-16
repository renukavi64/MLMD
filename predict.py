import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
def load-model(model_path="best_model.joblib"):
  model=joblib.load(model_path)
  return model
def make_prediction(model,input-data):
  prediction=model.predict(input_data)
  probabilisties=model.predict_proba(input_data)
  result="Maligant" if prediction[0]==0 else "Bengin"
  return{
    "predictions":result,
    "probabilisties":{
      "malignant":probabilisties[0][0],
      "bengin":probabilisties[0][1]
    }
  }
  if __name__==" __main__":
    model=load_model()
    data=load_breast_cancer()
    feature_names=data.feature_names
    sample_input=dict(zip(feature_names,data.data[0]))
    sample_input["mean radius"]=15.0
    sample_input[mean texture"]=18.0
    sample_input["mean perimeter"]=100.0
    input_df=pd.DataFrame(
      [sample_input],
      columns=feature_names
    )
    output=make_prediction(nodel.input_df)
    print("model loaded sucessfully!")
    print("\nPrediction:",output["prediction"])
    print("\nPrediction Probabilisties:")
    print("Malignant:",output["probabilisties"]["malignant"])
    print("Benign:",output["probabilisties"]["benign"])
  
  
