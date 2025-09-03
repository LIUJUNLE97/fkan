import torch
import torch.nn as nn
import torch.optim as optim
import os 
from torch.utils.data import DataLoader
from data.Datapair import MyDatasetNew, return_to_origin
from model.fkan import FourierKAN
from sklearn.preprocessing import StandardScaler
#from ..utils.Callback import TrainingLogger, make_dir
from utils.para_set import epochs, lr, batch_size, num_workers, beta_end
from sklearn.model_selection import train_test_split
import numpy as np 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data=np.load('data/p_1_5.npy')  # shape (N, C)

# load dataset
dataset = MyDatasetNew(data, seq_length=1024)

train_dataset, val_dataset = train_test_split(dataset, test_size=0.3, random_state=42)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = FourierKAN()

model_path = '/workspace/ljl/Pressure_forecasting/FKAN/results/fkan/checkpoint/checkpoint_epoch_400.pth'
checkpoint = torch.load(model_path, map_location='cpu')  # 加载整个checkpoint字典

model.load_state_dict(checkpoint['model_state_dict'])    # 加载模型权重

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
data_input = torch.tensor(val_dataset[50][0]).unsqueeze(0).to(device).float()  # 1, 26, 1000
output = model(data_input)
output = output.squeeze(0).cpu().detach().numpy()  # 1, 1, 1000, 26 -> 1000, 26
target_output = val_dataset[50][1]


scaler = StandardScaler()
scaler.fit_transform(data)  
pred = scaler.inverse_transform(output)  # 反标准化
targets = scaler.inverse_transform(target_output)  # 反标准化
np.save('results/fkan/pred_output_50.npy', pred)
np.save('results/fkan/target_output_50.npy', targets)