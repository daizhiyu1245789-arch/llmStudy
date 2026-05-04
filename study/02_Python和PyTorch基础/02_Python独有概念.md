# Python 独有概念（JavaScript 程序员需要特别注意）

> 这些是 JavaScript 里没有的概念，学的时候要格外留意。

---

## 1. 缩进决定代码块

Python 没有 `{}`，用**缩进**来划分代码块。

```python
# 缩进 = 4个空格（或1个 tab，但推荐空格）
if True:
    print("在 if 里面")   # 4个空格缩进
    if True:
        print("在嵌套 if 里面")  # 8个空格缩进
print("在 if 外面")        # 没缩进

# 错误示例（会导致 IndentationError）
# if True:
# print("没缩进")  ← 报错！
```

**这是最容易出错的地方**，IDLE / PyCharm / VSCode 会自动帮你缩进。

---

## 2. self = this

```python
class User:
    def __init__(self, name):
        self.name = name    # self.name = this.name

    def greet(self):
        return f"你好，{self.name}"   # self.name = this.name
```

JavaScript 里 this 是隐式的，Python 里 self 是显式的：
```js
// JavaScript
class User {
  constructor(name) {
    this.name = name;  // this 是隐式的
  }
  greet() {
    return `你好，${this.name}`;
  }
}
```

---

## 3. `__init__` = 构造函数

```python
class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

```js
// 等价的 JavaScript
class User {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }
}
```

---

## 4. 装饰器（@Decorator）

装饰器 = 函数的"包装纸"，类似 JavaScript 的高阶函数。

```python
# 定义一个装饰器
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("调用前")
        result = func(*args, **kwargs)
        print("调用后")
        return result
    return wrapper

# 使用装饰器（@ 开头）
@my_decorator
def say_hello(name):
    print(f"Hello, {name}")

# 等价于：
say_hello = my_decorator(say_hello)
```

本项目里装饰器的例子：
```python
# torch.no_grad() 是一个装饰器
@torch.no_grad()
def estimate_loss(model, batcher, *, eval_iters):
    # 在这里计算不需要梯度，节省显存
    ...
```

---

## 5. with 语句 = 自动资源清理

```python
# 打开文件，用完自动关闭（不需要手动 f.close()）
with open("test.txt", "r", encoding="utf-8") as f:
    content = f.read()
# 出了 with 块，f 自动关闭

# 等价于 JS 的 try-with-resources：
# try (const f = fs.openSync("test.txt")) {
#   const content = fs.readFileSync(f, "utf8");
# }
```

---

## 6. `*args` 和 `**kwargs`（可变参数）

```python
# *args = 接收任意数量的位置参数，变成元组
def sum_all(*numbers):
    print(numbers)   # (1, 2, 3)
    return sum(numbers)

sum_all(1, 2, 3)  # 6

# **kwargs = 接收任意数量的关键字参数，变成字典
def print_info(**info):
    print(info)   # {"name": "张三", "age": 28}

print_info(name="张三", age=28)

# 组合使用
def foo(*args, **kwargs):
    print(args)   # (1, 2)
    print(kwargs) # {"name": "王五"}
foo(1, 2, name="王五")
```

---

## 7. 切片赋值（强大）

```python
# 替换列表的一部分
nums = [0, 1, 2, 3, 4, 5]
nums[1:4] = [10, 20, 30]
print(nums)  # [0, 10, 20, 30, 4, 5]

# 快速清空
nums[:] = []
print(nums)  # []

# 二维切片
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(matrix[0][1:])   # [2, 3]（第一行的第2、3列）
print([row[0] for row in matrix])  # [1, 4, 7]（第一列）
```

---

## 8. 链式比较

```python
# Python 可以这样写
if 0 < x < 10:
    print("x 在 0 和 10 之间")

# JS 需要这样写：
# if (0 < x && x < 10)
```

---

## 9. 命名规范对照

```python
# Python 的 PEP8 规范（事实标准）
snake_case        # 变量、函数、方法：user_name, get_user()
PascalCase        # 类名：UserProfile, DataLoader
SCREAMING_SNAKE_CASE  # 常量：MAX_SIZE, PI

# 私有变量（约定俗成，以 _ 开头）
_private_var = 5    # 类似 JS 的 #private_var（但 Python 只是约定，不是语法）
```

---

## 10. True / False 的判断

```python
# 这些都是 False
bool(None)         # False
bool(0)            # False
bool("")           # False
bool([])           # False（空列表）
bool({})           # False（空字典）
bool(set())        # False（空集合）

# 其他都是 True
bool("hello")      # True
bool([1, 2])       # True
bool({"a": 1})     # True

# in 操作符
if "name" in {"name": "张三", "age": 28}:
    print("有 name 键")

if 3 in [1, 2, 3, 4, 5]:
    print("在列表里")
```

---

## 11. 列表 / 字典 / 集合 的推导式

```python
# 列表推导式（最常用）
squares = [x**2 for x in range(10)]

# 带条件的列表推导式
evens = [x for x in range(20) if x % 2 == 0]

# 字典推导式
word_len = {word: len(word) for word in ["apple", "banana"]}
# {'apple': 5, 'banana': 6}

# 集合推导式
unique_chars = {char for char in "hello"}
# {'h', 'e', 'l', 'o'}

# 生成器表达式（省内存，惰性求值）
sum(x**2 for x in range(1000000))  # 不需要建一个 100 万的列表
```

---

## 12. pass 语句（空操作占位）

```python
# Python 不允许空代码块，会报错
if True:
    # 还没想好怎么写
    pass  # 先写 pass 占位，不报错

# 等价于 JS:
# if (true) {
#   // TODO
# }
```

---

## 13. 链式操作

```python
# 可以这样连续调用（不需要括号换行）
result = (
    data
    .filter(lambda x: x > 0)
    .map(lambda x: x * 2)
    .reduce(lambda a, b: a + b, 0)
)
```

---

## 14. f-string = 模板字符串的进化版

```python
name = "张三"
age = 28

# f-string（Python 3.6+）
f"我叫{name}，今年{age}岁"    # 推荐用这个

# format（Python 2/3 兼容）
"我叫{}，今年{}岁".format(name, age)
"我叫{0}，今年{1}岁".format(name, age)
"我叫{n}，今年{a}岁".format(n=name, a=age)

# % 格式化（像 printf）
"我叫%s，今年%d岁" % (name, age)
```
