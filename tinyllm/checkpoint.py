"""
checkpoint.py - 模型断点的保存、加载与恢复

checkpoint（检查点）是训练过程中的"存档点"，包含：
- 模型权重（model_state_dict）
- 模型配置（GPTConfig，asdict 序列化后的字典）
- 优化器状态（optimizer_state_dict，包含学习率动量等信息）
- 当前训练步数（step）
- 历史上最佳的验证集损失（best_val_loss）
- 额外信息（extra，如分词器配置）

使用 checkpoint 可以：
- 中断训练后从断点恢复（resume），避免前功尽弃
- 保存最佳模型（best.pt），用于后续推理
- 定期保存中间状态（step_*.pt），便于分析和选择
"""

# 启用"未来"版本特性
from __future__ import annotations

# dataclasses.asdict：将数据类实例转换为字典（用于序列化）
from dataclasses import asdict

# Path：跨平台路径操作
from pathlib import Path

# Any：动态类型注解
from typing import Any

# PyTorch：张量序列化和反序列化
import torch

# ---- 本项目内部模块 ----
# 导入模型定义（避免循环导入）
from .model import GPT, GPTConfig


# ============================================================
# 1. 保存 checkpoint
# ============================================================

def save_checkpoint(
    path: str | Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    best_val_loss: float | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    将模型状态、优化器状态和训练元数据保存到磁盘。

    参数:
        path: checkpoint 文件保存路径（可以是字符串或 Path 对象）
        model: 要保存的 GPT 模型实例
        optimizer: PyTorch 优化器实例（用于恢复训练，可以为 None）
        step: 当前训练步数（1-indexed，即已完成多少步）
        best_val_loss: 历史上最佳的验证集损失（用于判断是否更新 best.pt）
        extra: 额外信息字典，通常包含分词器配置（tokenizer_to_extra 的返回值）

    保存的 checkpoint 结构：
        {
            "step": int,               # 当前步数
            "best_val_loss": float,    # 最佳验证损失
            "model_cfg": dict,         # GPTConfig 的字典形式
            "model_state": dict,       # model.state_dict()，所有模型权重
            "optimizer_state": dict,   # optimizer.state_dict()，优化器状态
            "extra": dict,             # 额外信息（分词器等）
        }

    注意：
        - path.parent.mkdir(parents=True, exist_ok=True) 会递归创建父目录
        - torch.save() 使用 pickle 序列化，会将张量以二进制形式保存到 .pt 文件
    """
    # ---- 构建 checkpoint 字典 ----
    ckpt = {
        # 当前训练步数（强制转为 int，避免 numpy int 等类型）
        "step": int(step),

        # 历史上最佳的验证集损失（用于 resume 时判断是否需要更新 best.pt）
        "best_val_loss": best_val_loss,

        # ---- 模型配置 ----
        # 将 GPTConfig 数据类实例转换为普通字典
        # asdict() 会递归地将所有字段转换为字典形式
        # 例如：GPTConfig(vocab_size=256, block_size=128, ...) -> dict(...)
        "model_cfg": asdict(model.cfg),

        # ---- 模型权重 ----
        # state_dict() 返回一个有序字典，包含所有可学习参数：
        # - 所有 Linear 层的 weight 和 bias
        # - 所有 Embedding 层的 weight
        # - 所有 LayerNorm 的 scale 和 bias（如果使用 bias）
        # 不包含 buffers（如 causal_mask，因为 persistent=False）
        "model_state": model.state_dict(),

        # ---- 优化器状态 ----
        # state_dict() 返回优化器的内部状态（学习率动量、方差估计等）
        # 如果 optimizer 为 None，则保存 None（断点可能不含优化器，如纯推理模型）
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,

        # ---- 额外信息 ----
        # extra 通常包含分词器配置，用于恢复训练时重建分词器
        # 如果没有额外信息，保存空字典（而不是 None，便于后续处理）
        "extra": extra or {},
    }

    # ---- 写入磁盘 ----
    path = Path(path)  # 确保是 Path 对象
    path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录（如 checkpoints/）
    torch.save(ckpt, path)  # 序列化保存到 .pt 文件


# ============================================================
# 2. 加载 checkpoint
# ============================================================

def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """
    从磁盘加载 checkpoint。

    参数:
        path: checkpoint 文件的路径
        map_location: 张量映射到的设备。
            - 默认 "cpu"：加载到 CPU（兼容性最好）
            - "cuda:0"：加载到第一块 GPU
            - torch.device("cuda:0")：同上
            注意：这里只影响张量的设备属性，不影响存储格式

    返回:
        checkpoint 字典，包含所有保存的字段（见 save_checkpoint 的返回结构）

    底层原理：
        torch.load() 使用 pickle 反序列化，会将 .pt 文件中的二进制数据
        重新构造为 Python 对象（字典、张量等）
    """
    return torch.load(path, map_location=map_location)


# ============================================================
# 3. 从 checkpoint 重建模型
# ============================================================

def build_model_from_checkpoint(
    ckpt: dict[str, Any],
    *,
    device: torch.device
) -> GPT:
    """
    根据 checkpoint 中的配置和权重重建完整的 GPT 模型。

    此函数封装了常见的"加载模型"流程：
    1. 从 checkpoint["model_cfg"] 提取超参数，创建 GPTConfig
    2. 用配置实例化一个新的 GPT 模型
    3. 将 checkpoint["model_state"] 中的权重加载到模型
    4. 将模型移动到指定设备

    参数:
        ckpt: 已加载的 checkpoint 字典
        device: 模型最终所在的设备（cpu 或 cuda）

    返回:
        权重已加载的 GPT 模型实例（处于 eval 模式）

    用途:
        - generate.py 中加载预训练模型进行推理
        - train.py 中从断点恢复训练
    """
    # ---- 从 checkpoint 恢复模型配置 ----
    # GPTConfig(**ckpt["model_cfg"])：
    # 将 checkpoint 中保存的字典作为关键字参数传给 GPTConfig 构造函数
    # 例如：GPTConfig(vocab_size=256, block_size=128, n_layer=4, ...)
    cfg = GPTConfig(**ckpt["model_cfg"])

    # ---- 实例化模型 ----
    model = GPT(cfg).to(device)

    # ---- 加载模型权重 ----
    # strict=True：确保 checkpoint 中的每个键都能在模型中找到对应参数，
    #             如果有缺失的键会抛出错误（帮助发现不匹配的 checkpoint）
    model.load_state_dict(ckpt["model_state"], strict=True)

    # 返回已加载权重的模型
    return model
