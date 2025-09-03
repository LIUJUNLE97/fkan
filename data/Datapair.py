import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. 定义 Dataset
# -------------------------
class MyDatasetNew(Dataset):
    def __init__(self, data, seq_length):
        """
        data: 已经标准化后的 numpy 数组, shape (N, C)
        seq_length: 时间序列长度
        """
        self.data = data
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)  
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - 2 * self.seq_length

    def __getitem__(self, idx):
        input_data = self.data[idx : idx + self.seq_length, :]      # (T, C)
        output_data = self.data[idx + self.seq_length : idx + 2*self.seq_length, :]
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)


# -------------------------
# 2. 数据预处理 (fit scaler 在整个 dataset 上)
# -------------------------
def prepare_datasets(raw_data, seq_length, train_ratio=0.8):
    """
    raw_data: 原始数据 numpy array, shape (N, C)
    seq_length: 时间序列长度
    train_ratio: 训练集比例
    """
    # (a) fit scaler on the whole dataset
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(raw_data)

    # (b) 划分 train/test
    N = len(data_scaled)
    split_idx = int(N * train_ratio)
    train_data = data_scaled[:split_idx]
    test_data  = data_scaled[split_idx:]

    # (c) 包装 Dataset
    train_dataset = MyDatasetNew(train_data, seq_length)
    test_dataset  = MyDatasetNew(test_data, seq_length)

    return train_dataset, test_dataset, scaler

def return_to_origin(raw_data, data_scaled):
    """
    raw_data: 原始数据 numpy array, shape (N, C)
    data_scaled: 标准化后的数据 numpy array, shape (M, C)
    """
    scaler = StandardScaler()
    scaler.fit(raw_data)  # fit on original data
    data_original = scaler.inverse_transform(data_scaled)
    return data_original

'''
# -------------------------
# 3. 使用示例
# -------------------------
if __name__ == "__main__":
    # 假设 raw_data shape (N, C)，这里用随机数举例
    N, C = 20000, 26
    raw_data = np.random.randn(N, C)

    seq_length = 1024
    train_dataset, test_dataset, scaler = prepare_datasets(raw_data, seq_length, train_ratio=0.8)

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # batch 检查
    x, y = next(iter(train_loader))
    print("x shape:", x.shape)  # (batch_size, seq_length, C)
    print("y shape:", y.shape)

    # -------------------------
    # 4. 预测结果 inverse_transform
    # -------------------------
    # 假设模型预测输出 (batch_size, seq_length, C)
    y_pred = y.numpy().reshape(-1, C)  # flatten (B*T, C)
    y_pred_original = scaler.inverse_transform(y_pred)
    y_pred_original = y_pred_original.reshape(y.shape[0], y.shape[1], C)  # (B, T, C)

    print("Recovered shape:", y_pred_original.shape)
'''