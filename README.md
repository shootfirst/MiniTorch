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

实现Scalar类运算符重载

简单

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

### 总结与分析

本章实现了标量的自动微分，涉及到的类有：ScalarHistory、Scalar、ScalarFunction、Context

- Scalar：标量，可以理解为跟踪记录最近一次函数处理信息的Scalar

- ScalarHistory：存储对当前Scalar操作的历史，包含ScalarFunction和Context

- ScalarFunction：用于操作上面的Scalar变量的数学函数，用于被不同数学函数继承

- Context：存储前向传播需要记录的值，如Task1.2中提到

接下来以正向构建计算图，存储函数信息和反向传播，计算自动微分两个流程，串联起自动微分的实现细节：以表达式𝑒=(𝑎+𝑏)∗(𝑏+1)为例，自动微分理论参见：https://fancyerii.github.io/books/autodiff/

#### 正向构建计算图

+ 计算a + b

+ 执行a.__add__(b)

+ a.__add__(b) 内部调用 Add.apply(a, b)

+ 执行add的前向传播函数，存储执行历史信息，新建ScalarHistory，返回执行的结果Scalar（见下面apply函数）

+ a + b执行的返回结果和b + 1的返回结果继续执行乘法，流程和上面一样，这样计算图被构建出来


、、、
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)
、、、


#### 反向传播实现自动微分

反向传播建立在上面构建好的计算图上，从最右节点e开始，参见上面的backpropagate函数实现

## module2

这一个实现主要是实现Tensor，即张量，前两章实现的是标量Scalar。但是深度学习归根结底是建立在张量之上。

### Tasks 2.1: Tensor Data - Indexing

张量的底层实际上就是一维数组加各个维度的信息，首先在此解释下：

Storage: 底层存储张量数据的一维数组
OutIndex: 输出下标
Index: 下标
Shape: 张量维度信息
Strides: 每一个维度的步长

该任务是实现两个函数：index_to_position，即把下标转换为底层存储的一维坐标

、、、
def index_to_position(index: Index, strides: Strides) -> int:
    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride
    return position
、、、

to_index，把底层存储的一维坐标转换为下标

、、、
def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh
、、、

### Tasks 2.2: Tensor Broadcasting

首先理解下张量的广播机制：

当一对张量满足下面的条件时，它们才是可以被“广播”的。

1、每个张量至少有一个维度。
2、迭代维度尺寸时，从尾部（也就是从后往前）开始，依次每个维度的尺寸必须满足以下之一：
    a、相等。
    b、其中一个张量的维度尺寸为1。
    c、其中一个张量不存在这个维度。

该任务是实现两个函数：broadcast_index，将可以广播的一对tensor，index长度较大tensor的index转换为较小那个tensor的index

、、、
def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0
、、、

shape_broadcast，返回一对tensor广播后的tensor的shape

、、、
def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    a, b = shape1, shape2
    m = max(len(a), len(b))
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    for i in range(m):
        if i >= len(a_rev):
            c_rev[i] = b_rev[i]
        elif i >= len(b_rev):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if c_rev[i] != a_rev[i] and a_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")
            if c_rev[i] != b_rev[i] and b_rev[i] != 1:
                raise IndexingError(f"Broadcast failure {a} {b}")

    return tuple(reversed(c_rev))
、、、

### Tasks 2.3: Tensor Operations

该task是完成tensor的所有函数，和标量类似，只是函数作用在tensor的每个元素上。最底层最基本的三个函数是map、zip和reduce

、、、
def tensor_map(fn: Callable[[float], float]) -> Any:

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index: Index = np.zeros(MAX_DIMS, np.int16)
        in_index: Index = np.zeros(MAX_DIMS, np.int16)
        for i in range(len(out)):
            # 得到对应的out_index
            to_index(i, out_shape, out_index)
            # 通过广播机制确定in_index
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # 通过index确定在存储的下标
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return _map
、、、

、、、
def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index: Index = np.zeros(MAX_DIMS, np.int16)
        a_index: Index = np.zeros(MAX_DIMS, np.int16)
        b_index: Index = np.zeros(MAX_DIMS, np.int16)
        for i in range(len(out)):
            # 得到对应的out_index
            to_index(i, out_shape, out_index)
            # 通过广播机制确定a_index和b_index
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # 通过index确定在存储的下标
            o = index_to_position(out_index, out_strides)
            a = index_to_position(a_index, a_strides)
            b = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[a], b_storage[b])

    return _zip
、、、


、、、
def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index: Index = np.zeros(MAX_DIMS, np.int16)
        reduce_size = a_shape[reduce_dim]
        for i in range(len(out)):
            # 得到对应的out_index
            to_index(i, out_shape, out_index)
            # 通过index确定在存储的下标
            o = index_to_position(out_index, out_strides)
            for s in range(reduce_size):
                out_index[reduce_dim] = s
                j = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a_storage[j])

    return _reduce
、、、

使用这三个最基本的函数实现张量的其他函数操作

和标量实现类似

、、、
class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output
、、、

、、、
class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output
、、、

实现tensor类对运算符的重载

类似

### Tasks 2.4: Gradients and Autograd

2.3贴的代码已经实现了张量的反向传播

### 总结

本章实现了标量的自动微分，涉及到的类有：

Tensor同Scalar一样，实现了Variable接口，所以反向传播的代码和Scalar一样。










