"""
data.py - 数据管道：文本语料处理、训练/验证集切分、随机批次采样
"""

# 启用"未来"版本特性
from __future__ import annotations

# dataclass 装饰器用于定义配置数据类
from dataclasses import dataclass

# NumPy：高效的数值计算库，用于存储和操作 token ID 数组
import numpy as np

# PyTorch：张量计算和自动求导
import torch


# ============================================================
# 1. 数据配置数据类
# ============================================================

@dataclass(frozen=True)
class DataConfig:
    """
    数据管道的配置参数。

    frozen=True 表示该数据类实例创建后不可修改（不可变对象），
    避免训练过程中意外修改配置。

    属性:
        block_size: 每个样本的上下文窗口长度。
            模型看到 block_size 个 token，预测第 block_size+1 个 token。
            也称为"context length"或"sequence length"。
        batch_size: 每个训练步（step）处理的样本数量。
            越大训练越快，但对显存要求越高。
        train_split: 训练集占总数据的比例（默认 0.9，即 90% 训练，10% 验证）。
    """
    block_size: int      # 上下文窗口大小（模型一次能"看到"的 token 数）
    batch_size: int      # 每个 step 的批大小
    train_split: float = 0.9  # 训练集占比（0.9 = 90% 训练，10% 验证）


# ============================================================
# 2. 文本批次采样器（TextBatcher）
# ============================================================

class TextBatcher:
    """
    文本批次采样器：负责将整个语料切分为训练/验证集，并按批次提供数据。

    工作流程：
    1. 接收一个一维的 token ID 数组（整个语料库）
    2. 按 train_split 比例切分为训练集和验证集
    3. get_batch() 每次随机从指定集合中采样一个批次

    数据表示方式（自回归语言模型的标准做法）：
    - 输入 x：连续 block_size 个 token
    - 目标 y：同一序列向后偏移 1 位（预测下一个 token）
      例如 x = [t0, t1, t2, ..., t_{k-1}]，y = [t1, t2, t3, ..., t_k]
    """

    def __init__(self, ids: np.ndarray, *, cfg: DataConfig, device: torch.device):
        """
        初始化 TextBatcher。

        参数:
            ids: 整个语料库的 token ID，形状 [total_tokens]，是 1D 数组
            cfg: DataConfig 配置对象
            device: PyTorch 设备（cpu 或 cuda:0），用于将张量放到对应设备
        """
        # ---- 输入合法性检查 ----
        # 确保 ids 是一维数组（语料库必须是扁平化的 token 序列）
        if ids.ndim != 1:
            raise ValueError("ids must be a 1D array")

        # 确保语料库足够大，至少能组成一个样本（block_size + 1 个 token）
        if len(ids) <= cfg.block_size + 1:
            raise ValueError(
                f"语料库太小：共 {len(ids)} 个 token，"
                f"但需要至少 {cfg.block_size + 1} 个 token 才能组成一个样本。"
            )

        n = len(ids)           # 语料库总 token 数
        min_len = cfg.block_size + 2  # 有效分割的最小长度要求

        # ---- 计算训练集/验证集的分割点 ----
        if n < min_len:
            # 语料库太小，无法进行有效分割
            raise ValueError("dataset too small for chosen block_size")

        if n < 2 * min_len:
            # 语料库较小（但足够一个样本），不进行分割，全部作为训练集
            split = n
        else:
            # 正常情况：按 train_split 比例分割
            split = int(n * cfg.train_split)

            # 确保分割后每部分都有至少 min_len 个 token
            # max(split, min_len)：训练集至少 min_len 个 token
            # min(..., n - min_len)：验证集也至少 min_len 个 token
            split = max(split, min_len)
            split = min(split, n - min_len)

        # ---- 切分训练集和验证集 ----
        self._train = ids[:split]  # 训练集：前 split 个 token
        self._val = ids[split:] if split < n else ids  # 验证集：剩余的 token

        # 保存配置和设备信息
        self.cfg = cfg
        self.device = device

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        随机采样一个批次的数据。

        参数:
            split: 数据集标识，"train" 表示从训练集采样，"val"/"val" 表示从验证集采样

        返回:
            x: 输入张量，形状 [batch_size, block_size]，类型 torch.int64
            y: 目标张量，形状 [batch_size, block_size]，类型 torch.int64
               y[i] 是 x[i] 对应的下一个 token（偏移 1 位）
        """
        # 根据 split 参数选择对应的数据集
        data = self._train if split == "train" else self._val

        # ---- 随机采样起始位置 ----
        # 有效起始位置范围：[0, max_start]，其中 max_start = len(data) - block_size - 1
        # 确保从每个起始位置开始的 block_size+1 个 token 都落在 data 范围内
        # 这样 y 的最后一个 token 也能取到（y = x 偏移 1 位）
        max_start = len(data) - (self.cfg.block_size + 1)
        if max_start <= 0:
            raise ValueError(
                f"数据集（{split}）太小，无法组成 block_size={self.cfg.block_size} 的样本。"
            )

        # 随机生成 batch_size 个起始位置
        # np.random.randint(0, max_start, size=(batch_size,))
        # 从 [0, max_start) 范围内均匀随机选择整数
        starts = np.random.randint(0, max_start, size=(self.cfg.batch_size,))

        # ---- 构建批次数据 ----
        # 对每个起始位置 s，提取：
        #   x = data[s : s + block_size]       （输入序列）
        #   y = data[s+1 : s+1 + block_size]   （目标序列，向后偏移 1）
        x = np.stack(
            [data[s: s + self.cfg.block_size] for s in starts],
            axis=0
        )
        y = np.stack(
            [data[s + 1: s + 1 + self.cfg.block_size] for s in starts],
            axis=0
        )

        # ---- 转换为 PyTorch 张量并放到指定设备 ----
        # np.int64 -> torch.int64（PyTorch 的标准整数类型）
        # non_blocking=True 启用异步数据传输（如果设备是 CUDA，可以减少数据传输等待时间）
        x_t = torch.from_numpy(x.astype(np.int64)).to(self.device, non_blocking=True)
        y_t = torch.from_numpy(y.astype(np.int64)).to(self.device, non_blocking=True)

        return x_t, y_t
