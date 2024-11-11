# MiniTorch笔记

## Module0

### Task 0.1: Operators

该任务是完成minitorch/operators.py文件中的数学函数，唯一需要注意的是log函数的实现，为了防止向下溢出，输入的参数变为0，对x加一个极小的浮点值1e-6

、、、
EPS = 1e-6

def log(x):
    return math.log(x + EPS)
、、、

### Task 0.2: Testing and Debugging

给任务1写的函数写单元测试，没有特别注意的细节，对函数特性进行测试即可

### Task 0.3: Functional Python

完成map、zip和reduce三个基本函数

、、、
def map(func):
     return lambda list: [func(x) for x in list]
、、、

、、、
def zipWith(func):
    return lambda list1, list2: [func(x, y) for x, y in zip(list1, list2)]
、、、

map 函数用于对序列中的每个项目应用一个函数
、、、
def reduce(func, start):
    def _reduce(func, list, start):
        iterator = iter(list)
        for i in iterator:
            start = func(start, i)
        return start
    return lambda list: _reduce(func, list, start)
、、、

通过三个函数实现negList、addLists、sum、prod

、、、
def negList(list):
    return map(neg)(list)
、、、

、、、
def addLists(list1, list2):
    return zipWith(add)(list1, list2)
、、、

、、、
def sum(list):
    return reduce(add, 0)(list)
、、、

、、、
def prod(list):
    return reduce(mul, 1)(list)
、、、

### Task 0.4: Modules

实现module类。在 PyTorch 中，torch.nn.Module 是构建神经网络的基本单元类。所有的模型、层，以及自定义模块，都需要继承自 torch.nn.Module。它提供了一个易于使用和高度灵活的框架来创建复杂的深度学习模型。
主要函数：
__init__: 构造函数，在这里定义层和参数。
forward: 定义前向传播逻辑。
parameters(): 返回模型所有参数的迭代器，通常用于优化器。
to(device): 将模块及参数移动到指定设备（如 GPU）。
eval(), train(): 设置模块为评估模式或训练模式。

实现train、eval、named_parameters和parameters。主要是named_parameters的实现

、、、
    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents.

        Returns
        -------
            The name and `Parameter` of each ancestor parameter.

        """

        def _named_parameters(module, prefix=""):
            for name, param in module._parameters.items():
                yield (prefix + name, param)
            for name, param in module._modules.items():
                yield from _named_parameters(param, prefix + name + ".")

        return list(_named_parameters(self))
、、、










