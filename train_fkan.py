import torch
import torch.nn as nn
import torch.optim as optim
import os 
from torch.utils.data import DataLoader
from data.Datapair import MyDatasetNew, return_to_origin
from model.fkan import FourierKAN
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

fkan = FourierKAN().to(device)
optimizer = optim.Adam(fkan.parameters(), lr=lr)


torch.set_num_threads(8)
from train.trainer import train_model, make_dir
base_dir = make_dir()
train_model(model=fkan, train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=epochs, beta=beta_end, optimizer=optimizer, base_dir=base_dir, resume_path=None, loss_type='frequency')