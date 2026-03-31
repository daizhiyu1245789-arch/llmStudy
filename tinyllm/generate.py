"""
generate.py - 基于训练好的 GPT 模型进行文本生成

功能：
- 从 checkpoint 加载预训练模型和分词器
- 使用自定义提示词（prompt）作为生成起点
- 通过温度采样（temperature）和 Top-K 过滤控制生成质量
- 将生成的 token ID 解码为可读文本并输出
"""

# 启用"未来"版本特性
from __future__ import annotations

# 命令行参数解析
import argparse

# sys 模块：访问 Python 运行时信息（如 stdout）
import sys

# PyTorch 深度学习框架
import torch

# ---- 本项目内部模块 ----
# checkpoint：模型加载工具
from .checkpoint import build_model_from_checkpoint, load_checkpoint

# tokenizer：分词器（从 checkpoint extra 中恢复）
from .tokenizer import tokenizer_from_extra


# ============================================================
# 1. 设备辅助函数
# ============================================================

def _get_device(device_arg: str) -> torch.device:
    """
    根据命令行参数解析 PyTorch 计算设备。

    参数:
        device_arg: 设备字符串，如 "auto"、"cpu"、"cuda"、"cuda:0"

    返回:
        torch.device 对象
    """
    if device_arg == "auto":
        # 自动检测：优先使用 CUDA GPU，如果不可用则回退到 CPU
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


# ============================================================
# 2. 主生成函数
# ============================================================

def main() -> None:
    """
    文本生成入口函数。

    完整流程：
    1. 解析命令行参数
    2. 加载 checkpoint（包含模型权重、分词器配置等）
    3. 重建模型和分词器
    4. 将提示词编码为 token ID
    5. 调用模型的 generate() 方法进行自回归生成
    6. 将生成的 token ID 解码为文本并打印
    """
    # ---- 配置 stdout 编码 ----
    # 确保输出能够正确处理 Unicode 字符（如中文）
    # 如果 stdout 不支持 errors 参数（某些环境），则静默跳过
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    # ---- 命令行参数解析 ----
    p = argparse.ArgumentParser(description="基于 GPT 模型生成文本")

    # checkpoint 路径（必需）
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="模型 checkpoint 文件的路径（.pt 格式）"
    )

    # 初始提示词
    p.add_argument(
        "--prompt",
        type=str,
        default="",
        help="生成文本的起始提示词（prompt）"
    )

    # 最大生成 token 数
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="最多生成多少个新 token（默认 200）"
    )

    # 温度参数（控制随机性）
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "采样温度。1.0 保持原始分布，"
            ">1.0 增加随机性，<1.0 增加确定性（默认 1.0）"
        )
    )

    # Top-K 过滤参数
    p.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="只保留概率最高的 k 个 token 进行采样（0 表示不过滤，默认 50）"
    )

    # 计算设备
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理设备：'auto'、'cpu'、'cuda' 等（默认 auto）"
    )

    args = p.parse_args()

    # ---- 设备设置 ----
    device = _get_device(args.device)

    # ---- 加载 checkpoint ----
    # torch.load 会将 checkpoint 中保存的所有张量加载到指定设备
    ckpt = load_checkpoint(args.ckpt_path, map_location=device)

    # ---- 恢复分词器 ----
    # 分词器信息存储在 checkpoint 的 extra 字段中
    tokenizer = tokenizer_from_extra(ckpt.get("extra"))

    # ---- 重建模型 ----
    # 根据 checkpoint 中保存的模型配置和权重恢复完整的 GPT 模型
    model = build_model_from_checkpoint(ckpt, device=device)

    # ---- 编码提示词 ----
    prompt_ids = tokenizer.encode(args.prompt)

    # 如果提示词为空（没有任何字符），用一个占位 token 代替
    if len(prompt_ids) == 0:
        prompt_ids = [0]

    # ---- 转换为 PyTorch 张量 ----
    # 形状：[batch_size=1, seq_len=len(prompt_ids)]
    # 确保张量在正确的设备上（CPU/CUDA）
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # ---- 自回归生成 ----
    # 模型会基于 idx 逐步生成新 token，直到生成 max_new_tokens 个为止
    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        # top_k <= 0 时表示不使用 top_k 过滤
        top_k=args.top_k if args.top_k > 0 else None,
    )

    # ---- 解码生成的 token 序列 ----
    # out 的形状：[batch_size=1, total_seq_len]
    # .tolist() 将张量转换为 Python 整数列表
    # tokenizer.decode 将 token ID 列表转换为字符串
    text = tokenizer.decode(out[0].tolist())

    # ---- 输出结果 ----
    print(text)


# ============================================================
# 3. 脚本入口
# ============================================================

# 当直接运行此脚本时（python -m tinyllm.generate）执行 main()
if __name__ == "__main__":
    main()
