# Rshadow optimizer
(c)2022-2024 Jean-Philippe Rasigade, <jean-philippe.rasigade@univ-lyon1.fr>, University of Lyon, Hospices Civils de Lyon, France.

This repository only holds pre-compiled binaries. The `Rshadow` package is the self-contained `R` frontend for the C++ **SHADOw** library. The source code is currently under embargo but will become open-source 'soon'. Feel free to use the pre-compiled binaries for any non-commercial purpose. 

*Commercial use is currently not permitted. Sorry about that ;-)*

## What does it do?

**SHADOw** stands for Sparse Hessian AutoDiff Optimization frameWork. It implements a high-performance second-order optimizer with automatic differentiation and operator overloading. To use `Rshadow`,

- declare an objective function, typically the log-likelihood of some statistical model, 
- let the package handle optimization, typically to find the maximum likelihood estimate of the model parameters
- obtain confidence intervals for the estimated parameters

That's it. You define the objective, the package searches for it.

## Install (Windows only)

```R
install.packages("https://github.com/rasigadelab/Rshadow/raw/main/Rshadow_0.1.zip", repos = NULL)
```

## Rshadow Hello World

`Rshadow` is an *operator overloading* package. The base **SHADOw** objects are the *tape*, which records operations in a computation graph, and the *spies*, which mimic basic R variables (numeric vectors, matrices, arrays). The spies behave as R objects but they spy upon the functions that we apply on them and record the functions on the tape. Once the tape has been recorded, the underlying C++ framework handles the optimization tasks using its own compiler.

Rshadow is targeted at maximum likelihood estimation, so we usually *maximize* the objective function. The basic steps of optimization are:
1. Declare an objective function using `objective`
2. Declare parameters of interest using `spy`
3. Write down the objective function with usual R syntax
4. Call `maximize`
5. Recover parameter values from the tape using `read`

Our *Hello World* example finds the maximum of the function $f(x) = -x^2$:
```r
library(Rshadow)
tape <- objective()
x <- spy(1.5, tape)
y <- -x^2
result <- maximize(tape)
stopifnot(as.numeric(x) == 1.5)
stopifnot(read(x, result) == 0)
```

## State of development

`Rshadow` uses an efficient, user-friendly operator overloading approach: the framework understands any function as long as it is part of the implementation (see available functions below). User-declared functions need not refer to any specific class. For instance, a user-defined function
```r
squared_diff <- function(x, y) (x - y)^2
```
will behave as expected independent of whether `x` and/or `y` are constants or free parameters.

### Available operators and functions

All functions apply on vectors and matrices unless stated otherwise.
- Operators `+`, `-`, `*`, `/`, `^`
- Comparison operators `<`, `<=`, `>`, `>=`
- Dot product `dot(x, y)`, matrix multiplication `x %*% y`
- Aggregation function, `sum`, `sumsq(x)`
- Unary functions `log`, `log1p`, `exp`, `lgamma`, `logit`, `logistic`

## Alternatives

**SHADOw** is under active development and should not be used for mission-critical and production tasks. 

For general-purpose optimization, consider [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/). For specialized autodiff and function manipulation, consider [JAX](https://jax.readthedocs.io/en/latest/index.html) or [CasADi](https://web.casadi.org/). For simple optimization over low-dimensional parameter spaces, use the built-in R method `optim`. For exhaustive exploration of complicated, possibly multimodal posterior distributions, consider using Bayesian frameworks such a [Stan](https://mc-stan.org/) or [BEAST2](https://www.beast2.org/).

## Going further? Read the docs

[Basic tutorial](tutorials/tutorial1_basics.md) showing simple regression and confidence intervals.