from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from init_model import ChangeModel
import torch

app = Flask(__name__)
tickers = ['MOEX','SBER','MGNT']
models = {'MOEX':'moex_model.pt','MGNT':'mgnt_model.pt','SBER':'sber_model.pt'}
features = ['tradedate', 'tradetime', 'secid', 'pr_open', 'pr_high', 'pr_low',
       'pr_close', 'pr_std', 'vol', 'val', 'trades', 'pr_vwap', 'pr_change',
       'trades_b', 'trades_s', 'val_b', 'val_s', 'vol_b', 'vol_s', 'disb',
       'pr_vwap_b', 'pr_vwap_s', 'SYSTIME']

def prepare_data(df:pd.DataFrame):
  df = df.drop(['SYSTIME','secid','tradedate','tradetime'],axis=1)
  df.fillna(df.mean(), inplace = True)
  df['target'] = df['pr_close'] - df['pr_open']
  df_numpy = df.to_numpy()
  sc = MinMaxScaler(feature_range=(-1,1))
  df_numpy_scaled =  sc.fit_transform(df_numpy)
  df_tensor = torch.tensor([df_numpy_scaled]).float()
  return df_tensor

@app.route("/api-data",methods=["POST"])
def get_data():
    if(request.method == "POST"):
       for ticker in tickers:
        json_data = request.get_json(ticker)
        if(ticker in json_data):
            json_data = list(json_data[ticker])
            df= pd.DataFrame([])
            for i in range(len(features)):
                df.at[0,features[i]] = json_data[i]
            input = prepare_data(df)
            model = torch.load('./pretrained_models/'+models[ticker])
            model.eval()
            with torch.no_grad():
                out = {'ticker':ticker, 'target': model(input).item()}
                return jsonify(out)

if __name__ == '__main__':
   app.run()