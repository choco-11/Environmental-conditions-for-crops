
import numpy as np
from flask import Flask, request, render_template
import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")
listofcrops = np.unique(df["crop"])
df[['temp', 'humidity','rainfall']] = df[['temp', 'humidity','rainfall']].apply(pd.to_numeric)

mean_df = {}
for crop in listofcrops:
  df_crop = df[ df['crop'] == crop]
  df_crop_opt = df_crop[df_crop['label'] == 'optimal']
  new_optimal = df_crop_opt.iloc[:,1:-1].reset_index().drop(["index"], axis=1)
  df_crop_not_opt = df_crop[df_crop['label'] == 'not optimal']
  new_not_optimal = df_crop_not_opt.iloc[:,1:-1].reset_index().drop(["index"], axis=1)

  mean_opt = np.array(new_optimal.mean(axis=0))
  mean_not_opt = np.array(new_not_optimal.mean(axis=0))
  
  mean_df[crop + "_optimal"] = mean_opt
  mean_df[crop + "_not_optimal"] = mean_not_opt

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    data = [x for x in request.form.values()]
    input_crop = data.pop(0)
    numeric_data = [int(i) for i in data]
    dist_optimal =  np.linalg.norm(numeric_data -   mean_df[input_crop + "_optimal"] )
    dist_not_optimal =  np.linalg.norm(numeric_data  - mean_df[input_crop +'_not_optimal'] )
    
    if dist_optimal > dist_not_optimal:
        predicted_text = "Environmental Conditions are not optimal for {}".format(input_crop)
    else:
        predicted_text = "Environmental Conditions are optimal for {}".format(input_crop)

    return render_template('index.html', predict_text = predicted_text)

if __name__ == "__main__":
    app.run(debug = True)