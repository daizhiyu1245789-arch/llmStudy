# Python vs JavaScript 语法对比

> Python 和 JavaScript 语法非常像，这篇只讲它们的不同点。

---

## 1. 变量声明

```python
# Python：不用 let/const/var，直接赋值
x = 5
name = "张三"
is_valid = True      # 注意：True 首字母大写（JS 用 true）
is_false = False     # False 也是大写

# JavaScript：
# let x = 5;
# const name = "张三";
# const isValid = true;
```

**Python 的命名规范**：
- 变量名：`snake_case`（小写下划线）
- 类名：`PascalCase`（首字母大写）
- 常量：`SCREAMING_SNAKE_CASE`

---

## 2. 数据类型

```python
# None（等于 JS 的 null）
x = None

# 布尔值
True, False   # JS: true, false

# 字符串
name = "张三"
desc = '单引号也行'
multi = """多行
字符串"""   # JS: `多行字符串`

# 整数 / 浮点数
age = 28          # int
price = 19.99     # float

# 布尔运算
and  # JS: &&
or   # JS: ||
not  # JS: !
```

---

## 3. 列表（类似 JS 数组）

```python
nums = [1, 2, 3, 4, 5]

# 索引（JS 数组一样）
nums[0]   # 第一个
nums[-1]   # 最后一个（Python 特有，很方便）
nums[-2]   # 倒数第二个

# 切片（JS 没有，Python 特有）
nums[1:4]    # [2, 3, 4]  （start:stop，不含 stop）
nums[:3]     # [1, 2, 3]  （从头开始）
nums[2:]     # [3, 4, 5]  （到末尾）
nums[::2]    # [1, 3, 5]  （步长 2）
nums[::-1]   # [5, 4, 3, 2, 1]  （反转）

# 常用方法
nums.append(6)   # 追加到末尾（push）
nums.pop()        # 弹出最后一个
nums.insert(0, 0) # 插入到指定位置
nums.remove(3)   # 移除第一个出现的值（不是索引）
nums.sort()       # 排序
nums.reverse()    # 反转

# 长度
len(nums)  # JS: nums.length
```

---

## 4. 字典（类似 JS 对象）

```python
user = {
    "name": "张三",
    "age": 28,
    "city": "北京"
}

# 取值
user["name"]       # "张三"
user.get("gender") # None（key 不存在不报错）
user.get("gender", "未知")  # "未知"（提供默认值）

# 添加/修改
user["gender"] = "男"
user["age"] = 29

# 删除
del user["city"]

# 常用方法
user.keys()      # dict_keys(['name', 'age', 'gender'])
user.values()    # dict_values(['张三', 29, '男'])
user.items()     # dict_items([('name','张三'), ('age',29)])
"age" in user    # True（判断 key 是否存在）
```

---

## 5. 元组（Tuple）- Python 特有

```python
# 元组是不可变的列表，用圆括号
point = (3, 4)
x, y = point   # 解包赋值：x=3, y=4

# 函数返回多个值时用元组
def get_user():
    return "张三", 28   # 实际返回 (name, age)

name, age = get_user()
```

---

## 6. 条件判断

```python
age = 20

# if / elif / else（注意没有 switch）
if age < 18:
    print("未成年")
elif age < 65:
    print("成年人")
else:
    print("老年人")

# 三元表达式（Python 特有写法）
status = "成年" if age >= 18 else "未成年"
# 等价于 JS: const status = age >= 18 ? "成年" : "未成年";
```

---

## 7. 循环

```python
# for 循环（Python 没有 i++，用 range 代替）
for i in range(5):        # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 6):     # 2, 3, 4, 5
    print(i)

for i in range(0, 10, 2): # 0, 2, 4, 6, 8（步长2）
    print(i)

# for 遍历列表
fruits = ["苹果", "香蕉", "橙子"]
for fruit in fruits:
    print(fruit)

# for 遍历字典
for key in user:
    print(key, user[key])

for key, value in user.items():
    print(f"{key}: {value}")

# while 循环（和 JS 一样）
i = 0
while i < 5:
    print(i)
    i += 1

# 列表推导式（Python 特有，很常用）
squares = [x**2 for x in range(10)]   # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
evens = [x for x in range(20) if x % 2 == 0]  # 偶数列表
```

---

## 8. 函数

```python
# 定义函数
def greet(name, age=18):   # 默认参数（JS 没有）
    return f"我叫{name}，今年{age}岁"

greet("张三")              # 用默认 age=18
greet("李四", age=30)      # 显式传参

# *args 和 **kwargs（可变参数）
def foo(*args, **kwargs):
    print(args)   # (1, 2, 3)      元组
    print(kwargs)  # {"name": "王五"} 字典

foo(1, 2, 3, name="王五")

# 匿名函数（类似 JS 箭头函数）
square = lambda x: x ** 2
square(5)  # 25

nums = [1, 2, 3, 4, 5]
list(map(lambda x: x * 2, nums))  # [2, 4, 6, 8, 10]
list(filter(lambda x: x % 2 == 0, nums))  # [2, 4]
```

---

## 9. 类

```python
class User:
    # 构造函数
    def __init__(self, name, age):
        self.name = name    # self = this
        self.age = age

    # 实例方法
    def introduce(self):
        return f"我叫{self.name}，{self.age}岁"

    # 类方法（类似 JS 的 static）
    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["age"])

    # 静态方法（类似普通函数，但属于这个类）
    @staticmethod
    def is_valid_age(age):
        return 0 <= age <= 150

# 创建实例
user = User("张三", 28)
print(user.introduce())

# 继承
class Student(User):
    def __init__(self, name, age, school):
        super().__init__(name, age)  # 调用父类构造函数
        self.school = school
```

---

## 10. 导入模块

```python
# 导入整个模块
import torch
torch.tensor([1, 2, 3])

# 导入特定内容（类似解构）
from torch import tensor, nn
tensor([1, 2, 3])
nn.Linear

# 导入并起别名
import torch.nn.functional as F
from pathlib import Path as P

# 相对导入（项目内部模块）
from . import model          # 同级目录
from .model import GPT       # 同级目录的 model.py
from .. import utils         # 上级目录
```

---

## 11. 常用内置函数

```python
# 类型转换
int("5")          # "5" → 5
float("3.14")     # "3.14" → 3.14
str(123)          # 123 → "123"
bool(0)           # False（0/""/None/[] 是 False，其他 True）

# 类型判断
type(5) == int              # True
isinstance(5, (int, float)) # True（int 或 float）

# 字符串操作
"hello".upper()           # "HELLO"
"hello".replace("l", "x") # "hexxo"
"hello world".split(" ")  # ["hello", "world"]
"-".join(["a", "b", "c"]) # "a-b-c"
```

---

## 12. 快速对照表

| JavaScript | Python | 说明 |
|------------|--------|------|
| `let x = 5` | `x = 5` | 变量，无关键字 |
| `const x = 5` | 变量不加关键字 | Python 没有 const |
| `true / false` | `True / False` | 首字母大写 |
| `null` | `None` | 空值 |
| `array.length` | `len(array)` | 长度 |
| `arr.push(6)` | `arr.append(6)` | 追加 |
| `Object.keys(obj)` | `dict.keys()` | 获取键 |
| `for (let i=0; i<5; i++)` | `for i in range(5)` | 循环 |
| `for (const item of arr)` | `for item in arr` | 遍历 |
| `arr.filter(fn)` | `[x for x in arr if fn(x)]` | 过滤 |
| `arr.map(fn)` | `[fn(x) for x in arr]` | 映射 |
| `arr.reduce(fn)` | `functools.reduce(fn, arr)` | 聚合 |
| `function fn(){}` | `def fn():` | 函数定义 |
| `() => x` | `lambda x: x` | 匿名函数 |
| `obj?.prop` | `obj.get("prop")` | 安全取值 |
| `import {x} from "m"` | `from m import x` | 导入 |
| `class A extends B{}` | `class A(B):` | 继承 |
