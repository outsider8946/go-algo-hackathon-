import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"
class ChangeModel(nn.Module):
  def __init__(self,input_size, hidden_size,num_stacked_size):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_size = num_stacked_size
    self.lstm = nn.LSTM(input_size,hidden_size,num_stacked_size,batch_first=True)
    self.lin = nn.Linear(hidden_size,1)
  def forward(self,x):
    batch = x.size(0)
    h0 = torch.zeros(self.num_stacked_size,batch,self.hidden_size).to('cpu')
    c0 = torch.zeros(self.num_stacked_size,batch,self.hidden_size).to('cpu')
    out,_ =self.lstm(x,(h0,c0))
    out = self.lin(out[:,-1,:])
    return out