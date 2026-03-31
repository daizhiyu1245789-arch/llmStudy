"""
tokenizer.py - 文本分词器：字符级和字节级两种实现

分词（Tokenization）是 NLP 的第一步：将原始文本转换为模型能处理的整数 ID 序列。

- CharTokenizer（字符级）：将每个字符映射为一个 ID，词表 = 所有不同字符的集合
- ByteTokenizer（字节级）：将每个字节（0-255）映射为一个 ID，词表固定为 256

字符级的优点：生成可读的文本，词表通常较小
字节级的优点：可以处理任意二进制/乱码文本，词表固定为 256
"""

# 启用"未来"版本特性
from __future__ import annotations

# dataclass 装饰器
from dataclasses import dataclass


# ============================================================
# 1. 分词器协议（接口约定）
# ============================================================

class TokenizerProtocol:
    """
    分词器的抽象接口（协议）。

    定义了所有分词器必须实现的 encode() 和 decode() 方法。
    使用 Python 的"协议"（Protocol）概念，这是一种静态类型的接口约定。

    属性:
        vocab_size: 词表大小，即不同 token 的总数
    """
    vocab_size: int  # 词表大小（子类需要定义为 @property 或属性）

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为 token ID 列表。

        参数:
            text: 输入文本字符串

        返回:
            token ID 列表，每个 ID 是 [0, vocab_size-1] 范围内的整数
        """
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """
        将 token ID 列表解码为文本字符串。

        参数:
            ids: token ID 列表

        返回:
            解码后的文本字符串
        """
        raise NotImplementedError


# ============================================================
# 2. 字节级分词器（ByteTokenizer）
# ============================================================

@dataclass(frozen=True)
class ByteTokenizer(TokenizerProtocol):
    """
    字节级分词器。

    核心思想：将 UTF-8 编码后的每个字节（8 bits，值域 0-255）作为一个 token。
    由于 UTF-8 是一种变长编码：
    - ASCII 字符（0-127）用 1 个字节表示
    - 中文等非 ASCII 字符用 2-4 个字节表示

    优点：
    - 词表固定为 256，不需要训练即可使用
    - 可以处理任意文本，包括乱码、二进制数据
    - 跨语言能力更强（所有语言都用相同的字节集合表示）

    缺点：
    - 序列长度通常比字符级更长（中文每个字符对应 2-4 个字节）
    """
    vocab_size: int = 256  # 字节级词表大小固定为 256（0-255）

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为字节级 token ID 列表。

        过程：文本 -> UTF-8 字节序列 -> 字节值列表（0-255）

        参数:
            text: 输入文本字符串

        返回:
            每个字节对应的 ID 列表（0-255 的整数）
        """
        # str.encode("utf-8")：将 Unicode 字符串转换为 UTF-8 字节序列
        # errors="replace"：遇到无法编码的字符用 ?（U+FFFD）替换，避免崩溃
        data = text.encode("utf-8", errors="replace")

        # bytes 对象可以直接索引，每个元素是 0-255 的整数
        # 转换为 list[int] 以符合接口约定
        return list(data)

    def decode(self, ids: list[int]) -> str:
        """
        将字节级 token ID 列表解码为文本字符串。

        过程：字节值列表（0-255）-> UTF-8 字节序列 -> 文本字符串

        参数:
            ids: 每个元素是 0-255 范围的整数

        返回:
            解码后的文本字符串
        """
        # 将整数列表转换为 bytes 对象
        # int(i) & 0xFF 确保每个值在 0-255 范围内（防止负数或过大值）
        data = bytes(int(i) & 0xFF for i in ids)

        # bytes.decode("utf-8")：将 UTF-8 字节序列转换回字符串
        # errors="replace"：遇到无效的 UTF-8 序列用 ? 替换，避免解码失败
        return data.decode("utf-8", errors="replace")


# ============================================================
# 3. 字符级分词器（CharTokenizer）
# ============================================================

@dataclass(frozen=True)
class CharTokenizer(TokenizerProtocol):
    """
    字符级分词器。

    核心思想：将文本中的每个唯一字符作为一个 token，词表 = 所有不同字符的集合。

    编码：char -> index（在 itos 中的位置）
    解码：index -> char（从 itos 中查找）

    特殊 token：
    - unk_token（<unk>）：未知字符的占位符，用于处理训练时未见过的字符

    优点：
    - 序列长度短（每个字符一个 token，对中文尤其友好）
    - 生成可读的中间结果

    缺点：
    - 词表取决于训练语料，出现新字符时需要用 unk_token 处理
    """
    itos: tuple[str, ...]      # index-to-string：索引到字符的映射表（列表），e.g., ("<unk>", "a", "b", ...)
    unk_token: str = "<unk>"  # 未知字符的替代符号

    @property
    def vocab_size(self) -> int:
        """
        返回词表大小。
        """
        return len(self.itos)

    @classmethod
    def train(cls, text: str, *, unk_token: str = "<unk>") -> "CharTokenizer":
        """
        根据给定文本训练一个新的字符级分词器。

        过程：
        1. 统计文本中出现的所有不同字符
        2. 构建 itos 映射表（索引 -> 字符），unk_token 放在第一个位置

        参数:
            text: 训练语料文本
            unk_token: 未知字符的替代符号，默认为 "<unk>"

        返回:
            训练好的 CharTokenizer 实例
        """
        # sorted(set(text))：获取所有不同字符并排序（排序是为了保证确定性和可复现性）
        chars = sorted(set(text))

        # 如果 unk_token 恰好在字符集中，移除它（unk_token 单独占一个索引）
        if unk_token in chars:
            chars.remove(unk_token)

        # 构建 itos：unk_token 在前，后面是所有字符
        # 第一个索引（0）固定留给 unk_token
        itos = (unk_token, *chars)

        # 返回一个不可变的 CharTokenizer 实例
        return cls(itos=tuple(itos), unk_token=unk_token)

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为字符级 token ID 列表。

        参数:
            text: 输入文本字符串

        返回:
            每个字符对应的 ID 列表
        """
        # 构建字符串到索引的映射表（stoi = string-to-index）
        stoi = {ch: i for i, ch in enumerate(self.itos)}

        # unk_token 对应的 ID（如果不存在则为 0，即 unk_token 的位置）
        unk_id = stoi.get(self.unk_token, 0)

        # 对每个字符查找对应 ID，未知字符则用 unk_id
        return [stoi.get(ch, unk_id) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """
        将字符级 token ID 列表解码为文本字符串。

        参数:
            ids: token ID 列表

        返回:
            拼接后的文本字符串
        """
        out = []  # 用于存储解码后的字符

        for i in ids:
            # 确保索引在有效范围内（0 <= i < vocab_size）
            if 0 <= int(i) < len(self.itos):
                out.append(self.itos[int(i)])  # 正常字符
            else:
                out.append(self.unk_token)      # 超出范围的 ID 用 unk_token 替代

        return "".join(out)  # 将字符列表拼接为单个字符串


# ============================================================
# 4. 分词器的序列化/反序列化（用于 checkpoint 保存和加载）
# ============================================================

def tokenizer_to_extra(tokenizer: TokenizerProtocol) -> dict:
    """
    将分词器对象序列化为字典（存放在 checkpoint 的 extra 字段中）。

    参数:
        tokenizer: 分词器实例（ByteTokenizer 或 CharTokenizer）

    返回:
        一个可以放入 checkpoint extra 字段的字典

    异常:
        TypeError: 如果遇到不支持的分词器类型
    """
    if isinstance(tokenizer, ByteTokenizer):
        # 字节级分词器只需要记录类型（不需要额外参数，词表固定为 256）
        return {"type": "byte"}

    if isinstance(tokenizer, CharTokenizer):
        # 字符级分词器需要记录类型、词表映射表（itos）和未知字符符号
        return {
            "type": "char",
            "itos": list(tokenizer.itos),     # tuple -> list（JSON 序列化需要）
            "unk_token": tokenizer.unk_token
        }

    # 不支持的分词器类型
    raise TypeError("unsupported tokenizer")


def tokenizer_from_extra(extra: dict | None) -> TokenizerProtocol:
    """
    从 checkpoint 的 extra 字段反序列化分词器对象。

    参数:
        extra: checkpoint 中保存的 extra 字典。
               如果为 None 或空字典，返回默认的 ByteTokenizer。

    返回:
        恢复的分词器实例（ByteTokenizer 或 CharTokenizer）
    """
    # 容错处理：如果 extra 为 None 或空
    if not extra:
        return ByteTokenizer()  # 默认使用字节级分词器

    # 从 extra 中获取分词器配置
    tok = extra.get("tokenizer")
    if tok is None:
        return ByteTokenizer()  # 没有 tokenizer 信息，默认字节级

    # 根据类型字符串恢复对应分词器
    if tok.get("type") == "byte":
        # 字节级：词表固定，不需要额外参数
        return ByteTokenizer()

    if tok.get("type") == "char":
        # 字符级：需要还原 itos 映射表和 unk_token
        itos = tok.get("itos")
        # 验证 itos 是有效的字符串列表
        if not isinstance(itos, list) or not all(isinstance(x, str) for x in itos):
            raise ValueError("invalid char tokenizer vocab in checkpoint")
        # 还原为 tuple（原始格式），并获取 unk_token（默认为 "<unk>"）
        return CharTokenizer(
            itos=tuple(itos),
            unk_token=str(tok.get("unk_token", "<unk>"))
        )

    # 未知的类型，默认返回字节级分词器
    return ByteTokenizer()
