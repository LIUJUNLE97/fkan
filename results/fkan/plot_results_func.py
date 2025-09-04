# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Times New Roman'
import numpy as np
import matplotlib.ticker as ticker
def plot_ins_snap_Cp(pred, target, snapid):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
    Cp_pred = pred[snapid, :]
    Cp_target = target[snapid, :]
    x = np.arange(1, 27, 1)  # 空间点数
    ax.plot(x, Cp_pred, linestyle='-', marker= 'o', label='Forecast', color='blue')
    ax.plot(x, Cp_target, linestyle='-.', marker = 's', label='Experimental data', color='orange')
    ax.set_xlabel('Pressure tap No.', fontname='Times New Roman', fontsize=16)
    ax.set_ylabel(r'$C_p$', fontname='Times New Roman', fontsize=16)
    ax.set_xlim([0.5, 26.5])
    # 图例设置
    ax.legend(prop={'family': 'Times New Roman', 'size': 14})
    # 刻度与字体设置
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)
    # 网格线样式
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    # 坐标轴边框线条加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    fig.savefig(f'Cp_plot_snapid_{snapid}.pdf', format='pdf', bbox_inches='tight')
    
def plot_mean_std_cp(pred, target):
    """
    绘制每个压力测点的时间平均 Cp 值及标准差对比图（预测 vs 实验），并保存为 PDF。

    参数:
        pred (ndarray): shape (T, N)，预测值，N 为空间点数，T 为时间步数
        target (ndarray): shape (T, N)，真实实验值
        save_path (str): 保存路径，默认为当前目录的 'mean_std_plot.pdf'
    """
    # 计算均值和标准差（时间方向）
    mean_pred = np.mean(pred, axis=0)
    std_pred = np.std(pred, axis=0)
    mean_target = np.mean(target, axis=0)
    std_target = np.std(target, axis=0)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
    x = np.arange(1, 27)  # 空间点编号


    ax.errorbar(x, mean_pred, yerr=std_pred,
    fmt='o-', label='Forecast',
    capsize=3, elinewidth=2  # 加粗误差线
        )
    # Experimental data：误差线为虚线
    ax.errorbar(
    x, mean_target, yerr=std_target,
    fmt='s--', label='Experimental data',
    capsize=3, elinewidth=1.5, linestyle='--',  # 虚线误差线
    ecolor='orange',  # 可选：设置误差线颜色
    )


    ax.set_xlabel('Pressure tap No.', fontname='Times New Roman', fontsize=16)
    ax.set_ylabel(r'$\bar{C_p} \pm \sigma $', fontname='Times New Roman', fontsize=16)

    # 图例设置
    ax.legend(prop={'family': 'Times New Roman', 'size': 14})

    # 刻度与字体设置
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    # 网格线样式
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim([0.5, 26.5])  # 设置 x 轴范围

    # 坐标轴边框线条加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_path = 'mean_std_plot.pdf'  # 保存路径
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def plot_fluctuation_power_spectrum(pred, target, point_No, fs, nperseg):
    """
    绘制指定压力测点的**脉动功率谱密度**（预测 vs 实验），自动去除时间平均值，仅显示波动部分。
    
    参数:
        pred (ndarray): shape (T, N)，预测值，N 为空间点数，T 为时间步数
        target (ndarray): shape (T, N)，真实值
        point_No (int): 空间测点编号（从 0 开始）
        fs (float): 采样频率，默认 1.0 Hz
        nperseg (int): Welch 分段长度，默认 256
        save_path (str): 保存路径，默认 'fluctuation_spectrum_point_{point_No}.pdf'
    """
    # 去除均值，获取脉动部分
    pred_fluc = pred[:, point_No] - np.mean(pred[:, point_No])
    target_fluc = target[:, point_No] - np.mean(target[:, point_No])
    #pred_low_fluc = pred_low[:, point_No] - np.mean(pred_low[:, point_No])

    # Welch 方法计算脉动的功率谱密度
    f_pred, Pxx_pred = welch(pred_fluc, fs=fs, nperseg=nperseg)
    f_pred_plot = f_pred*0.003
    f_target, Pxx_target = welch(target_fluc, fs=fs, nperseg=nperseg)
    f_target_plot = f_target*0.003
    #f_pred_low, Pxx_pred_low = welch(pred_low_fluc, fs=fs, nperseg=nperseg)
    #f_pred_low_plot = f_pred_low*0.003

    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)

    ax.semilogy(f_pred_plot, Pxx_pred, label='Forecast', color='blue')
    ax.semilogy(f_target_plot, Pxx_target, label='Experimental data', color='orange')
    #ax.semilogy(f_pred_low_plot, Pxx_pred_low, label='Forecast (No Beta)', color='green')

    ax.set_xlabel(r'$S_t$', fontname='Times New Roman', fontsize=16)
    ax.set_ylabel('Fluctuating Power Spectral Density', fontname='Times New Roman', fontsize=16)
    ax.set_xlim([0, 0.5])

    # 图例与样式
    ax.legend(prop={'family': 'Times New Roman', 'size': 14})
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_path = f'fluctuation_spectrum_point_{point_No}.pdf'
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_cp(pred, target, point_No):
    """
    绘制每个压力测点的 Cp 值对比图（预测 vs 实验），并保存为 PDF。

    参数:
        pred (ndarray): shape (T, N)，预测值，N 为空间点数，T 为时间步数
        target (ndarray): shape (T, N)，真实实验值
        save_path (str): 保存路径，默认为当前目录的 'cp_plot.pdf'
    """
    # 绘图
    fig, ax= plt.subplots(figsize=(6, 4), dpi=600)
    
    x = np.arange(pred.shape[0])  # 压力测点编号

    # 绘制预测值
    ax.plot(x, pred[:, point_No], linewidth=1, linestyle='-', label='Forecast')

    # 绘制实验值
    ax.plot(x, target[:, point_No], linewidth=0.5, linestyle='-.', label='Experimental data')
    ax.set_xlim([0,1024])

    # 设置坐标轴标签
    ax.set_xlabel(r'$\Delta T / \delta t$', fontname='Times New Roman', fontsize=16)
    ax.set_ylabel(r'$C_p$', fontname='Times New Roman', fontsize=16)
    
    # 图例设置
    ax.legend(prop={'family': 'Times New Roman', 'size': 14})

    # 刻度与字体设置
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    # 网格线样式
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 坐标轴边框线条加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # plt.title('Mean Cp at Each Spatial Point')  # 可选标题
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save_path = f'cp_plot_point_{point_No}.pdf'  # 保存路径根据点编号动态生成
    # 保存图像为 PDF
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()  # 防止多次绘图干扰

def plot_power_spectrum(pred, target, point_No, fs, nperseg):
    """
    绘制每个压力测点的功率谱密度对比图（预测 vs 实验），并保存为 PDF。

    参数:
        pred (ndarray): shape (T, N)，预测值，N 为空间点数，T 为时间步数
        target (ndarray): shape (T, N)，真实实验值
        point_No (int): 压力测点编号
        save_path (str): 保存路径，默认为当前目录的 'power_spectrum_plot.pdf'
    """
    from scipy.signal import welch

    # 计算功率谱密度
    f_pred, Pxx_pred = welch(pred[:, point_No], fs=fs, nperseg=nperseg)
    f_target, Pxx_target = welch(target[:, point_No], fs=fs, nperseg=nperseg)

    # 绘图
    fig, ax = plt.subplots(figsize=(4, 3), dpi=600)
    
    ax.semilogy(f_pred, Pxx_pred, label='Forecast', color='blue')
    ax.semilogy(f_target, Pxx_target, label='Experimental data', color='orange')

    ax.set_xlabel('Frequency [Hz]', fontname='Times New Roman', fontsize=16)
    ax.set_ylabel('Power Spectral Density [dB/Hz]', fontname='Times New Roman', fontsize=16)

    # 图例设置
    ax.legend(prop={'family': 'Times New Roman', 'size': 14})

    # 刻度与字体设置
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    # 网格线样式
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 坐标轴边框线条加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    save_path = f'power_spectrum_plot_point_{point_No}.pdf'  # 保存路径根据点编号动态生成
    fig.savefig(save_path, format='pdf',  bbox_inches='tight')
    plt.close()  # 防止多次绘图干扰

def plot_cl(pred, target):
    """
    绘制每个压力测点的 Cl 值对比图（预测 vs 实验），并保存为 PDF。

    参数:
        pred (ndarray): shape (T, N)，预测值，N 为空间点数，T 为时间步数
        target (ndarray): shape (T, N)，真实实验值
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
    Cl_pred = np.sum((pred[:, 0:8] - pred[:, 13:21]), axis=1)/8
    Cl_target = np.sum((target[:, 0:8] - target[:, 13:21]), axis=1)/8
    time = np.arange(Cl_pred.shape[0])  # 时间步数
    ax.plot(time, Cl_pred, linestyle='-', label='Forecast', color='blue')
    ax.plot(time, Cl_target, linestyle='-.', label='Experimental data', color='orange')

    ax.set_xlabel(r'$\Delta T / \delta t$', fontname='Times New Roman', fontsize=16)
    ax.set_ylabel(r'$C_l$', fontname='Times New Roman', fontsize=16)
    ax.set_xlim([0, 1024])

    # 图例设置
    ax.legend(prop={'family': 'Times New Roman', 'size': 14})
    ax.set_ylim([-1.25, 1])

    # 刻度与字体设置
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    # 网格线样式
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 坐标轴边框线条加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    Cl_pred_std = np.std(Cl_pred)
    Cl_target_std = np.std(Cl_target)
    print(f'Cl_pred_std: {Cl_pred_std}, Cl_target_std: {Cl_target_std}')
    fig.savefig('Cl_plot.pdf', format='pdf', bbox_inches='tight')

def plot_cd(pred, target):
    """
    绘制每个压力测点的 Cd 值对比图（预测 vs 实验），并保存为 PDF。

    参数:
        pred (ndarray): shape (T, N)，预测值，N 为空间点数，T 为时间步数
        target (ndarray): shape (T, N)，真实实验值
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
    Cd_pred = np.abs(np.sum((pred[:, 8:13] - pred[:, 21:26]), axis=1)/5)
    Cd_target = np.abs(np.sum((target[:, 8:13] - target[:, 21:26]), axis=1)/5)
    time = np.arange(Cd_pred.shape[0])  # 时间步数
    ax.plot(time, Cd_pred, linestyle='-', label='Forecast', color='blue')
    ax.plot(time, Cd_target, linestyle='-.', label='Experimental data', color='orange')

    ax.set_xlabel(r'$\Delta T / \delta t$', fontname='Times New Roman', fontsize=16)
    ax.set_ylabel(r'$C_d$', fontname='Times New Roman', fontsize=16)
    ax.set_xlim([0, 1024])
    ax.set_ylim([1.3, 1.9])

    # 图例设置
    ax.legend(prop={'family': 'Times New Roman', 'size': 14})

    # 刻度与字体设置
    ax.tick_params(direction='in')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(14)

    # 网格线样式
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    # 坐标轴边框线条加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    Cd_pred_mean = np.mean(Cd_pred)
    Cd_target_mean = np.mean(Cd_target)
    print(f'Cd_pred_mean: {Cd_pred_mean}, Cd_target_mean: {Cd_target_mean}')
    fig.savefig('Cd_plot.pdf', format='pdf', bbox_inches='tight')
    