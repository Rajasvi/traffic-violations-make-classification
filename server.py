import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from utils import pre_process

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

feats = ['Issue Date', 'Issue time',
       'RP State Plate', 'Body Style',
       'Color', 'Agency', 'Violation code',
         "Route",'Fine amount', 'Latitude', 'Longitude']

cat_cols = ['RP State Plate', 'Body Style', 'Color', 'Agency', 'Violation code',"Route",
        'IssueHour', 'IssueWeek', 'IssueYear', 'IssueWeekDay', 'IssueDay']


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    df = pd.DataFrame([data])
    assert set(df.columns).intersection(set(feats))==set(feats), "Insufficient number of input features provided"
    x = pre_process(df[feats],cat_cols)

    pred_proba = model.predict_proba(x)
    pred = model.predict(x)[0]
    print(f"Prediction Prob.: {pred_proba}")
  
    return jsonify({"popular make probability":str(np.round(pred_proba[0][1],3)), "prediction":str(pred)})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
