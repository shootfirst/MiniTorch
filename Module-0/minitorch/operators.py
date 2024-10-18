"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul (x, y):
    return x * y

def id (x):
    return x

def add (x, y):
    return x + y

def neg(x):
    return float(-x)

def lt(x, y):
    return float(x < y)

def eq(x, y):
    return float(x == y)

def max(x, y):
    return x if x > y else y

def is_close(x, y):
    return abs(x - y) < 1e-2

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def relu(x):
    return max(0.0, x)

EPS = 1e-6

def log(x):
    return math.log(x + EPS)

def exp(x):
    return math.exp(x)

def log_back(x, d):
    return d / x

def inv(x):
    return 1.0 / x

def inv_back(x, d):
    return -d / x ** 2

def relu_back(x, d):
    return d if x > 0 else 0.0

def sigmoid_back(x, d):
    return d * exp(-x) / ((1 + exp(-x)) ** 2)






# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(func):
     return lambda list: [func(x) for x in list]

def zipWith(func):
    return lambda list1, list2: [func(x, y) for x, y in zip(list1, list2)]

def reduce(func, start):
    def _reduce(func, list, start):
        iterator = iter(list)
        for i in iterator:
            start = func(start, i)
        return start
    return lambda list: _reduce(func, list, start)

def negList(list):
    return map(neg)(list)

def addLists(list1, list2):
    return zipWith(add)(list1, list2)

def sum(list):
    return reduce(add, 0)(list)

def prod(list):
    return reduce(mul, 1)(list)
