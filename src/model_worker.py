import json
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ... как то получаем запрос на обработку данных  в  json формате объекта в форме X = {'secid':a,..} - обычная строка в дате 
# и храним в json_data (тип list)
json_data = list() #TODO


def prepare_data(json_data:list):
  df = pd.DataFrame(json_data)
  df = df.drop(['secid'])
  df.fillna(df.mean(), inplace = True)
  df['target'] = df['pr_close'] - df['pr_open']
  df_numpy = df.to_numpy()
  sc = MinMaxScaler(feature_range=(-1,1))
  return sc.fit_transform(df_numpy)

def work(json_data):
   model_name = None
   if(json_data['secid'] == 'SBER'):
     model_name = 'sber_model.pt'
   elif(json_data['secid'] == 'MOEX'):
     model_name = 'meox_model.pt'
   elif(json_data['secid'] == 'MGNT'):
     model_name = 'mgnt_model.pt'
   else:
     print('Wrong data to predict! Tradestat should be from SBER, MOEX or MGNT!')
     return None
   if(model_name!=None):
     model = torch.load('pretrained_models/'+model_name)
     model.eval()
     data = prepare_data(json_data)
     out = model(data)
     out = pd.DataFrame({'change':out}).to_json()
     # TODO далее отправка out на сервис go

work(json_data)

