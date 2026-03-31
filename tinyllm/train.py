"""
train.py - GPT 模型训练脚本

功能：
- 加载和分词文本语料
- 构建 GPT 模型
- 使用 AdamW 优化器训练（支持断点恢复、梯度裁剪）
- 周期性评估验证集损失
- 自动保存 checkpoint（最佳模型 + 最新模型 + 定期保存）
"""

# 启用"未来"版本特性
from __future__ import annotations

# 命令行参数解析
import argparse

# nullcontext：条件上下文管理器（amp 时用 autocast，否则用空上下文）
from contextlib import nullcontext

# Path：跨平台路径操作
from pathlib import Path

# time：计时工具（用于计算训练速度和打印日志）
from time import time

# NumPy：数值计算
import numpy as np

# PyTorch 深度学习框架
import torch

# tqdm：进度条可视化
from tqdm import tqdm

# ---- 本项目内部模块 ----
# checkpoint：模型断点保存/加载工具
from .checkpoint import load_checkpoint, save_checkpoint

# data：数据管道（TextBatcher）
from .data import DataConfig, TextBatcher

# model：GPT 模型定义
from .model import GPT, GPTConfig

# tokenizer：分词器（字符级和字节级）
from .tokenizer import ByteTokenizer, CharTokenizer, tokenizer_from_extra, tokenizer_to_extra


# ============================================================
# 1. 设备辅助函数
# ============================================================

def _get_device(device_arg: str) -> torch.device:
    """
    根据命令行参数解析 PyTorch 设备。

    参数:
        device_arg: 设备字符串，如 "auto"、"cpu"、"cuda"、"cuda:0" 等

    返回:
        torch.device 对象

    逻辑:
        - "auto"：自动检测，有 CUDA 则用 CUDA，否则用 CPU
        - 其他：直接使用指定的设备字符串
    """
    if device_arg == "auto":
        # torch.device：根据字符串创建设备对象
        # torch.cuda.is_available()：检测是否有可用的 CUDA GPU
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# ============================================================
# 2. 损失评估函数
# ============================================================

@torch.no_grad()  # 推理时不需要梯度，节省显存和计算量
def estimate_loss(
    model: GPT,
    batcher: TextBatcher,
    *,
    eval_iters: int
) -> dict[str, float]:
    """
    评估模型在训练集和验证集上的损失。

    参数:
        model: 待评估的 GPT 模型
        batcher: TextBatcher 数据批次采样器
        eval_iters: 评估时采样的批次数（取平均以减少随机波动）

    返回:
        包含 "train" 和 "val" 两个键的字典，值为平均损失值
    """
    model.eval()  # 切换到评估模式（关闭 Dropout 等训练特有的层）

    out: dict[str, float] = {}  # 存储评估结果

    # 对训练集和验证集分别评估
    for split in ("train", "val"):
        losses = []  # 存储每一批的损失值

        # 多次采样取平均，减小随机波动
        for _ in range(eval_iters):
            # 从指定数据集获取一个批次
            x, y = batcher.get_batch(split)

            # 前向传播：计算损失（第二个返回值）
            _, loss = model(x, y)

            # 将损失从张量转换为 Python float
            losses.append(float(loss.item()))

        # 计算平均损失
        out[split] = float(np.mean(losses))

    model.train()  # 切回训练模式
    return out


# ============================================================
# 3. 主训练函数
# ============================================================

def main() -> None:
    """
    训练入口函数。

    完整的训练流程：
    1. 解析命令行参数
    2. 设置随机种子（保证可复现性）
    3. 加载和预处理语料
    4. 构建/恢复模型和优化器
    5. 训练循环（采样 -> 前向传播 -> 反向传播 -> 参数更新）
    6. 周期性评估和保存 checkpoint
    """
    # ---- 命令行参数解析 ----
    p = argparse.ArgumentParser(description="训练 GPT 语言模型")

    # ---- 数据相关参数 ----
    p.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="训练语料文件的路径（必须是 .txt 文本文件）"
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default="char",
        choices=["char", "byte"],
        help="分词方式：'char' 字符级 或 'byte' 字节级（默认 char）"
    )

    # ---- 运行时参数 ----
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备：'auto'（自动检测）、'cpu'、'cuda'、'cuda:0' 等"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="随机种子，用于保证实验可复现性（默认 1337）"
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help="训练精度：'fp32'（32位浮点）或 'bf16'（BF16 混合精度）"
    )

    # ---- 模型架构参数 ----
    p.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="上下文窗口长度，即模型一次能处理的 maximum token 数"
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="每个训练 step 的批大小"
    )
    p.add_argument(
        "--n_layer",
        type=int,
        default=4,
        help="Transformer Block 的层数"
    )
    p.add_argument(
        "--n_head",
        type=int,
        default=4,
        help="多头注意力的头数"
    )
    p.add_argument(
        "--n_embd",
        type=int,
        default=128,
        help="Token 嵌入向量的维度"
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout 概率（防止过拟合）"
    )

    # ---- 优化器参数 ----
    p.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="AdamW 的初始学习率（默认 3e-4）"
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="AdamW 的权重衰减系数（正则化，默认 0.1）"
    )
    p.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="梯度裁剪阈值（防止梯度爆炸，0 表示不裁剪）"
    )

    # ---- 训练步数/日志/保存参数 ----
    p.add_argument(
        "--max_steps",
        type=int,
        default=2000,
        help="总共训练多少个 step（默认 2000）"
    )
    p.add_argument(
        "--eval_interval",
        type=int,
        default=200,
        help="每隔多少 step 评估一次验证集损失（默认 200）"
    )
    p.add_argument(
        "--eval_iters",
        type=int,
        default=50,
        help="每次评估时采样多少个批次取平均（默认 50）"
    )
    p.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="每隔多少 step 打印一次训练日志（默认 20）"
    )
    p.add_argument(
        "--save_interval",
        type=int,
        default=200,
        help="每隔多少 step 保存一次增量 checkpoint（默认 200）"
    )

    # ---- checkpoint 保存路径 ----
    p.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints",
        help="checkpoint 保存目录（默认 'checkpoints'）"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="是否从 latest.pt 断点恢复训练"
    )

    # 解析所有参数
    args = p.parse_args()

    # ---- 设置随机种子 ----
    # 确保 PyTorch 和 NumPy 的随机操作可复现
    torch.manual_seed(args.seed)   # PyTorch 全局随机种子
    np.random.seed(args.seed)       # NumPy 随机种子

    # ---- 设备设置 ----
    device = _get_device(args.device)  # 解析设备（CPU/CUDA）

    # ---- 启用 Tensor Core（CUDA 专用）----
    # 设置矩阵乘法精度为 "high" 可以启用 Tensor Core 加速（需要 Volta 架构或更新的 GPU）
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ---- 加载语料文本 ----
    # Path(...).read_text() 读取整个文件为字符串
    text = Path(args.data_path).read_text(encoding="utf-8", errors="replace")

    # ---- 创建 checkpoint 目录 ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # 如果目录不存在则创建
    latest_path = out_dir / "latest.pt"         # 最新的 checkpoint 路径

    # ---- 尝试加载断点（如果 resume=True 且存在）----
    resume_ckpt = None
    if args.resume and latest_path.exists():
        # load_checkpoint 从磁盘读取 checkpoint 并将张量映射到目标设备
        resume_ckpt = load_checkpoint(latest_path, map_location=device)

    # ---- 初始化分词器 ----
    if resume_ckpt is not None:
        # 断点恢复：从 checkpoint 中恢复分词器
        tokenizer = tokenizer_from_extra(resume_ckpt.get("extra"))
    else:
        # 新训练：根据参数选择字符级或字节级分词器
        if args.tokenizer == "char":
            # 字符级：需要从语料中训练（统计所有字符构建词表）
            tokenizer = CharTokenizer.train(text)
        else:
            # 字节级：词表固定为 256，不需要训练
            tokenizer = ByteTokenizer()

    # ---- 将文本转换为 token ID 数组 ----
    # tokenizer.encode(text)：文本 -> [0, vocab_size-1] 的整数列表
    # np.array(..., dtype=np.int32)：转换为 NumPy 数组（int32 节省内存）
    ids = np.array(tokenizer.encode(text), dtype=np.int32)

    # ---- 构建数据批次采样器 ----
    data_cfg = DataConfig(
        block_size=args.block_size,
        batch_size=args.batch_size
    )
    batcher = TextBatcher(
        ids,
        cfg=data_cfg,
        device=device
    )

    # ---- 构建模型配置 ----
    if resume_ckpt is not None:
        # 断点恢复：从 checkpoint 中恢复模型配置
        model_cfg = GPTConfig(**resume_ckpt["model_cfg"])
    else:
        # 新训练：根据参数或默认值构建配置
        model_cfg = GPTConfig(
            vocab_size=tokenizer.vocab_size,  # 词表大小（由分词器决定）
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )

    # ---- 实例化模型并移动到设备 ----
    model = GPT(model_cfg).to(device)

    # ---- 构建优化器（AdamW）----
    optimizer = torch.optim.AdamW(
        model.parameters(),              # 要优化的所有参数
        lr=args.learning_rate,          # 学习率
        weight_decay=args.weight_decay,  # 权重衰减（L2 正则化）
        betas=(0.9, 0.95),               # 动量估计的指数衰减率（Adam 超参数）
    )

    # ---- 断点恢复：加载模型和优化器状态 ----
    step = 0               # 当前 step（从 0 开始计数）
    best_val_loss = None   # 历史上最佳的验证集损失

    if resume_ckpt is not None:
        # 从 checkpoint 中恢复 step 数
        step = int(resume_ckpt.get("step", 0))
        # 恢复最佳验证损失
        best_val_loss = resume_ckpt.get("best_val_loss")
        # 加载模型权重（strict=True 确保所有键都匹配）
        model.load_state_dict(resume_ckpt["model_state"], strict=True)
        # 加载优化器状态（如学习率动量等）
        if resume_ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])

    # ---- 混合精度训练（AMP）----
    # 如果设备是 CUDA 且启用 bf16，则使用 PyTorch 的自动混合精度
    amp_dtype = None
    if args.dtype == "bf16" and device.type == "cuda":
        amp_dtype = torch.bfloat16  # BF16：比 FP16 更稳定，梯度范围更大

    # torch.amp.autocast：前向传播使用指定精度，反向传播自动适配
    # nullcontext：如果不需要 AMP，则使用空上下文（不起作用）
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype
        else nullcontext()
    )

    # ---- 训练循环 ----
    t0 = time()  # 记录训练开始时间（用于计算每秒处理多少 step）

    # tqdm：显示训练进度条
    pbar = tqdm(
        range(step, args.max_steps),  # 从 step 开始到 max_steps
        initial=step,                   # 初始 step（断点恢复时不为 0）
        total=args.max_steps            # 总 step 数（决定进度条百分比）
    )

    for step in pbar:
        # ---- 获取一个训练批次 ----
        x, y = batcher.get_batch("train")

        # ---- 梯度清零 ----
        # set_to_none=True：将梯度设置为 None 而不是 0（稍微节省显存）
        optimizer.zero_grad(set_to_none=True)

        # ---- 前向传播（可能使用混合精度）----
        with autocast_ctx:
            _, loss = model(x, y)

        # ---- 反向传播 ----
        # 计算损失对所有参数的梯度
        loss.backward()

        # ---- 梯度裁剪 ----
        # 防止梯度爆炸（常见于 RNN/Transformer 的早期训练阶段）
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip
            )

        # ---- 参数更新 ----
        optimizer.step()

        # ---- 日志输出 ----
        if (step + 1) % args.log_interval == 0:
            elapsed = time() - t0  # 已用时间（秒）
            # 更新进度条描述：显示当前 step、训练损失、已用时间
            pbar.set_description(
                f"step {step+1} loss {loss.item():.4f} ({elapsed:.1f}s)"
            )

        # ---- 周期性评估和保存 checkpoint ----
        if (step + 1) % args.eval_interval == 0 or (step + 1) == args.max_steps:
            # 评估训练集和验证集损失
            losses = estimate_loss(model, batcher, eval_iters=args.eval_iters)
            val_loss = losses["val"]

            # 如果当前验证损失优于历史最佳，则保存为 best.pt
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    out_dir / "best.pt",        # 保存路径
                    model=model,               # 模型实例
                    optimizer=optimizer,        # 优化器（包含学习率动量等状态）
                    step=step + 1,             # 当前 step（1-indexed）
                    best_val_loss=best_val_loss,  # 最佳验证损失
                    # 额外信息：包含分词器配置（用于恢复训练时重建分词器）
                    extra={"tokenizer": tokenizer_to_extra(tokenizer)},
                )

            # 同时保存为 latest.pt（始终保持一个可恢复的最新断点）
            save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                step=step + 1,
                best_val_loss=best_val_loss,
                extra={"tokenizer": tokenizer_to_extra(tokenizer)},
            )

        # ---- 周期性保存增量 checkpoint ----
        if (step + 1) % args.save_interval == 0:
            # 每 save_interval 步保存一个带 step 编号的 checkpoint
            save_checkpoint(
                out_dir / f"step_{step+1}.pt",
                model=model,
                optimizer=optimizer,
                step=step + 1,
                best_val_loss=best_val_loss,
                extra={"tokenizer": tokenizer_to_extra(tokenizer)},
            )


# ============================================================
# 4. 脚本入口
# ============================================================

# 当直接运行此脚本时（python -m tinyllm.train）执行 main()
# 导入为模块时不执行（防止运行训练脚本时意外触发训练）
if __name__ == "__main__":
    main()
