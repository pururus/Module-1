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


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: First float number.
        y: Second float number.

    Returns:
    -------
        Product x * y.

    """
    return x * y


def id(x: float) -> float:
    """Identity function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The input value unchanged.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        Sum x + y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        Negated value -x.

    """
    return -x


def lt(x: float, y: float) -> float:
    """Less-than comparison.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if x < y, otherwise False.

    """
    return x < y


def eq(x: float, y: float) -> float:
    """Equality comparison with tolerance.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if |x - y| < 1e-10, otherwise False.

    """
    return abs(x - y) < 1e-10


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        The larger of x and y.

    """
    return y if lt(x, y) else x


def is_close(x: float, y: float) -> float:
    """Check whether two numbers are close.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if |x - y| < 1e-2, otherwise False.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        Sigmoid of x, computed in a numerically stable way.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the rectified linear unit.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        max(x, 0.0).

    """
    return max(x, 0.0)


def log(x: float) -> float:
    """Compute the natural logarithm.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        Natural logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        e raised to the power x.

    """
    return math.exp(x)


def log_back(x: float, y: float) -> float:
    """Backward pass for logarithm.

    Args:
    ----
        x: Input value from forward pass.
        y: Upstream gradient.

    Returns:
    -------
        Gradient with respect to x.

    """
    return y / x


def inv(x: float) -> float:
    """Compute the multiplicative inverse.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        1.0 / x.

    """
    return 1.0 / x


def inv_back(x: float, y: float) -> float:
    """Backward pass for multiplicative inverse.

    Args:
    ----
        x: Input value from forward pass.
        y: Upstream gradient.

    Returns:
    -------
        Gradient with respect to x.

    """
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Backward pass for ReLU.

    Args:
    ----
        x: Input value from forward pass.
        y: Upstream gradient.

    Returns:
    -------
        y if x >= 0, otherwise 0.0.

    """
    if x >= 0:
        return y
    return 0.0


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


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Create a function that maps f over an iterable.

    Args:
    ----
        f: Function to apply.

    Returns:
    -------
        Function taking an iterable and returning mapped values.

    """

    def map_fn(l: Iterable[float]) -> Iterable[float]:
        return [f(x) for x in l]

    return map_fn


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Create a function that combines two iterables elementwise with f.

    Args:
    ----
        f: Binary function to apply.

    Returns:
    -------
        Function taking two iterables and returning zipped results.

    """

    def zip_fn(l1: Iterable[float], l2: Iterable[float]) -> Iterable[float]:
        return [f(x, y) for (x, y) in zip(l1, l2)]

    return zip_fn


def reduce(
    f: Callable[[float, float], float], init: float
) -> Callable[[Iterable[float]], float]:
    """Create a function that reduces an iterable using f and an initial value.

    Args:
    ----
        f: Binary reduction function.
        init: Initial accumulator value.

    Returns:
    -------
        Function taking an iterable and returning the reduced value.

    """

    def reduce_fn(l: Iterable[float]) -> float:
        res = init
        for x in l:
            res = f(x, res)

        return res

    return reduce_fn


def negList(l: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
        l: Iterable of numbers.

    Returns:
    -------
        Iterable of negated numbers.

    """
    return map(neg)(l)


def addLists(l1: Iterable[float], l2: Iterable[float]) -> Iterable[float]:
    """Add two lists elementwise.

    Args:
    ----
        l1: First iterable of numbers.
        l2: Second iterable of numbers.

    Returns:
    -------
        Iterable of pairwise sums.

    """
    return zipWith(add)(l1, l2)


def sum(l: Iterable[float]) -> float:
    """Sum all elements in an iterable.

    Args:
    ----
        l: Iterable of numbers.

    Returns:
    -------
        Sum of elements.

    """
    return reduce(add, 0.0)(l)


def prod(l: Iterable[float]) -> float:
    """Multiply all elements in an iterable.

    Args:
    ----
        l: Iterable of numbers.

    Returns:
    -------
        Product of elements.

    """
    return reduce(mul, 1.0)(l)
