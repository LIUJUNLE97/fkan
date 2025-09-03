def make_dir():
    import os
    dir = os.getcwd()

    folder = 'results'  # 运行后请记住
    base_dir = f'{dir}/{folder}/fkan'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        # log_path = f'{base_dir}/log.txt'
    else:
        i = 1
        while os.path.exists(f'{base_dir}_{i}'):
            i += 1
        base_dir = f'{base_dir}_{i}'
        os.makedirs(base_dir)
        # log_path = f'{base_dir}/log.txt'
    return base_dir
import torch
import torch.nn.functional as F
def frequency_loss_new(pred, target):
    """
    计算高于指定频率阈值部分的频域 MSE 损失（只针对时间维度 FFT）。
    
    参数:
        pred: [B, 1, T, S]，预测值（时间维度T=1000）
        target: [B, 1, T, S]，目标值
        sample_rate: 采样频率，例如 400Hz
        high_freq_threshold: 高频起始的频率阈值，例如 100Hz
    返回:
        高频频率段的 MSE 损失（频域模值）
    """
    # 时间维度长度
    sample_rate=400
    #high_freq_threshold=100
    T = pred.size(1)
    #freq_resolution = sample_rate / T  # 每一点代表的频率

    # 起始索引位置
    cutoff_idx = 220

    # FFT，按时间维度，输出 [B, T, S]
    pred_fft = torch.fft.fft(pred, dim=1)
    target_fft = torch.fft.fft(target, dim=1)

    # 频谱是对称的，取前半部分（实数信号）
    fft_len = T // 2 + 1
    pred_fft = pred_fft[:, :fft_len, :]
    target_fft = target_fft[:, :fft_len, :]

    # 高频裁剪
    pred_high = pred_fft[:, cutoff_idx:, :]
    target_high = target_fft[:, cutoff_idx:, :]

    # 计算模值的 MSE（也可替换为 real/imag 分别 MSE）
    pred_mag = torch.abs(pred_high)
    target_mag = torch.abs(target_high)

    loss = F.mse_loss(pred_mag, target_mag)
    return loss

def train_model(model, train_dataloader, val_dataloader, num_epochs, beta, optimizer, base_dir, resume_path, loss_type=None):
    import torch 
    import torch.nn as nn
    import os
    """
    训练模型
    :param model: 模型
    :param train_dataloader: 训练数据加载器
    :param val_dataloader: 验证数据加载器
    :param num_epochs: 训练轮数
    :param optimizer: 优化器
    :param base_dir: 模型保存路径
    :return: None
    """

    checkpoint_path = os.path.join(base_dir, "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)

    with open(f'{checkpoint_path}/fkan_train_log.txt', 'w') as d:
        d.write("Epoch\tTrain Loss\tVal Loss\n")

    # 定义变量以保存最小val_loss和对应的模型参数
        
        start_epoch = 0
        min_val_loss = float('inf')
        best_model_params = None

    # 如果指定了 checkpoint 路径，则加载
        if resume_path is not None and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            min_val_loss = checkpoint.get('min_val_loss', float('inf'))
            print(f"Resumed training from epoch {start_epoch}, min_val_loss: {min_val_loss:.4f}")

    
        train_criterion = nn.MSELoss()
        val_criterion = nn.MSELoss()

        device = torch.device("cuda:0")
        model = model.to(device)
    
        for epoch in range(num_epochs):
            model.train()
            dynamic_beta = beta * (epoch + 1) / num_epochs
            train_loss = 0.0
            total_samples = 0

            for input_data, target_data in train_dataloader:
                optimizer.zero_grad()
                inputs = input_data.float().to(device)
                targets = target_data.float().to(device)

                outputs = model(inputs)
                if loss_type == 'frequency':
                    #print('frequency loss is embedded')
                    loss = val_criterion(outputs, targets) + dynamic_beta * frequency_loss_new(outputs, targets)
                else:
                    loss = train_criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)


    # 切换到评估模式
            model.eval()
            val_loss = 0.0
            total_samples = 0

            with torch.no_grad():
                for input_data, target_data in val_dataloader:
                    inputs = input_data.float().to(device)  # 添加维度 bs, 26, 1000 -> bs, 1, 26, 1000
                    targets = target_data.float().to(device)  # 添加维度 bs, 26, 1000 -> bs, 1, 26, 1000

                    outputs = model(inputs)
                    if loss_type == 'frequency':
                        loss = val_criterion(outputs, targets) + dynamic_beta * frequency_loss_new(outputs, targets)
                    else:
                
                        loss = val_criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)

    # 计算平均训练损失和验证损失
            train_loss /= len(train_dataloader)
            val_loss /= len(val_dataloader)
    # 写入训练日志
            log_str = f"{epoch+1}\t{train_loss:.4f}\t{val_loss:.4f}\n"
            d.write(log_str)

    # 检查是否是最小的val_loss，并保存模型参数
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model_params = model.state_dict()
    # 调用学习率调度器
    #scheduler.step(val_loss)
    # 构建日志字典
            logs = {"train_loss": train_loss, "val_loss": val_loss}

    # 调用回调函数的方法:打印训练过程中保存的字典文件
            #callback.on_epoch_end(epoch, logs)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            # 每 200 轮保存一次 checkpoint
            if (epoch + 1) % 200 == 0:
                checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'min_val_loss': min_val_loss,
                }
                torch.save(checkpoint, f'{checkpoint_path}/checkpoint_epoch_{epoch+1}.pth')


    # 保存模型
        torch.save(best_model_params, f'{checkpoint_path}/fkan_best.pth')
