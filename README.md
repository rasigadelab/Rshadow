# Rshadow automatic optimizer for R

Rshadow is an automatic optimizer for large, sparse computational graphs such as those induced by probabilistic graphical models. The typical usage is maximum-likelihood estimation (MLE) over complex models. The name means <u>s</u>parse <u>H</u>essian <u>a</u>utomatic <u>d</u>ifferentiation <u>o</u>ptimization frame<u>w</u>ork.

Author and maintainer: Jean-Philippe Rasigade,
University of Lyon, <jean-philippe.rasigade@univ-lyon1.fr><br/>
Rshadow is licensed under CC BY - NC - ND 4.0, https://creativecommons.org/licenses/by-nc-nd/4.0/<br/>
Private/commercial licensing options available upon request.

## Installation and requirements

`Rshadow` requires Rtools and packages `RcppEigen` and `devtools` (for GitHub installation)

Install using `devtools::install_github("rasigadelab/Rshadow")`

Be patient (like, prepare coffee): C++ compilation may take several minutes.

## Rshadow Hello World

`Rshadow` is an *operator overloading* package. The base SHADOw objects are the *tape*, which records operations in a computation graph, and the *spies*, which mimic basic R variables (numeric vectors, matrices, arrays), spy upon the functions that we apply on them, and record the functions on the tape. Once the tape has been recorded, the underlying C++ framework handles the optimization tasks using its own compiler.

Rshadow is targeted at maximum likelihood estimation, so we usually *maximize* the objective function. The basic steps of optimization are:
1. Declare an objective function using `tape_new`
2. Declare parameters of interest using `spy`
3. Write down the objective function with the usual R syntax
4. Call `maximize`
5. Recover parameter values from the tape using `read`

Our *Hello World* example finds the maximum of the function $f(x) = -x^2$:
```
library(Rshadow)
tape <- tape_new()
x <- spy(1.5, tape)
y <- -x^2
result <- maximize(tape)
stopifnot(as.numeric(x) == 1.5)
stopifnot(read(x, result) == 0)
```

## Features

`Rshadow` features:
- efficient operator overloading that differentiates free parameters and fixed parameters. The framework understands any function that uses implemented operations. The functions needs not refer to any specific Shadow class. See file R/functions.R for examples.
- reverse-mode automatic 2nd-order differentiation. The autodiff algorithm computes the gradient and (sparse) Hessian in a single pass. See [Wang, Gebremedhin & Pothen 2016](https://link.springer.com/article/10.1007/s12532-016-0100-3) for details of Hessian computation algorithms.
- backtracking regularized Newton optimization. The optimizer is a variant of Newton's method. Computational efficiency results from using sparse solver algorithms when inverting the Hessian. Numerical stability relies on adaptive Tikhonov regularization. The optimizer performs a backtracking line search along the direction of the Newton iteration vector.

## State of development

### List of current implemented operators and functions

All functions apply on vectors and matrices unless stated otherwise.
- Operators `+`, `-`, `*`, `/`, `^`
- Comparison operators `<`, `<=`, `>`, `>=` (Iverson bracket behavior)
- Dot product `dot(x, y)`, matrix multiplication `x %*% y`
- Sum, `sum(x)`, sum of squares `sumsq(x)`
- Unary functions `log`, `log1p`, `exp`, `lgamma`, `logit`, `logistic`

## Alternatives

SHADOw is under development and should not be used for critical and production tasks. For general-purpose optimization, consider TensorFlow or PyTorch. For specialized autodiff and function manipulation, consider JAX. For a reference implementation of [Wang, Gebremedhin & Pothen 2016](https://link.springer.com/article/10.1007/s12532-016-0100-3), see [CSCsw/LivarH](https://github.com/CSCsw/LivarH) on GitHub.