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
        
        def _named_parameters(module, prefix=""):
            for name, param in module._parameters.items():
                yield (prefix + name, param)
            for name, param in module._modules.items():
                yield from _named_parameters(param, prefix + name + ".")

        return list(_named_parameters(self))
        
、、、


## Module1

### Task 1.1: Numerical Derivatives

实现数值微分

、、、
def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    vals1 = [v for v in vals]
    vals1[arg] += epsilon
    return (f(*vals1) - f(*vals)) / epsilon
、、、

### Task 1.2: Scalars

完成scalarFunction中各个子类Scalar函数的forward前向传播函数，关键点是需要存储关键变量来给反向传播去使用

Add加法函数，对加法求导为1，所以无需记录

、、、
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b
、、、

Log函数，需要存储传入相乘的系数，供反向传播使用

、、、
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)
、、、

Mul乘法函数，求导为乘法的系数，所以需要存储相乘的两个数

、、、
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a,b)
        return a * b
、、、

Neg函数，求导加上负号即可

、、、
    def forward(ctx: Context, a: float) -> float:
        return operators.neg(a)
、、、

Sigmoid、Relu、Exp同理

### Task 1.3: Chain Rule

实现链式法则chain_rule函数。首先我们需要理解几个类：ScalarHistory、Scalar、ScalarFunction、Context

- Scalar：用于自动微分的标量值。尽可能地表现为标准的Python数字，同时跟踪它的相关函数操作，只能通过ScalarFunction操作。

- ScalarHistory：存储对当前Scalar操作的历史，包含ScalarFunction和Context

- ScalarFunction：用于操作上面的Scalar变量的数学函数，用于被不同数学函数继承

- Context：存储前向传播需要记录的值，如Task1.2中提到

、、、
    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        return list(zip(h.inputs, x))
、、、


### Task 1.4: Backpropagation

1、首先需要实现所有的反向传播函数（1.2实现了正向传播），核心是使用正向传播时存储的上下文

Add加法函数，由于加法导数为1，直接返回当前的入参，由于加法有两个输入参数，所以需要返回两份

、、、
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output
、、、

Log函数，需要使用正向传播存储的系数

、、、
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)
、、、

Mul乘法函数，乘法导数为其相乘的系数

、、、
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return b * d_output, a * d_output
、、、

后面函数都是类似思路，不过有两个值得注意，LT和EQ导数都是0，入参都是两个，所以直接返回0

2、其次我们需要实现拓扑排序函数，这个就是leetcode原题，在此之前我们需要研究下Variable类



、、、
def topological_sort(variable: Variable) -> Iterable[Variable]:
    sort = []
    visited = set()

    def dfs(v: Variable):
        # if the node is already visited or constant, return
        if v.unique_id in visited or v.is_constant():
            return

        # otherwise, add the node to the visited set
        visited.add(v.unique_id)

        # recursively visit all the parents of the node
        for parent in v.parents:
            dfs(parent)

        # add the current node at the front of the result order list
        sort.insert(0, v)

    dfs(variable)

    return sort
、、、

3、最后实现反向传播backpropagate函数

、、、

def backpropagate(variable: Variable, deriv: Any) -> None:
    
    order = topological_sort(variable)

    # initialize derivatives dictionary
    gradients = {variable.unique_id: deriv}

    for v in order:
        # constant nodes do not contribute to the derivatives
        if v.is_constant():
            continue

        # derivative of the current tensor
        grad = gradients.get(v.unique_id)

        # chain rule to propogate to parents
        if not v.is_leaf():
            for parent, chain_deriv in v.chain_rule(grad):
                if parent.unique_id in gradients:
                    gradients[parent.unique_id] += chain_deriv
                else:
                    gradients[parent.unique_id] = chain_deriv

        # only accumulate derivatives for leaf nodes
        if v.is_leaf():
            v.accumulate_derivative(grad)
            
、、、








