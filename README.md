# Rshadow automatic optimization for R

Rshadow is an automatic optimizer for large, sparse computational graphs such as those induced by probabilistic graphical models. The typical usage is maximum-likelihood estimation (MLE) over complex models. The name means <u>s</u>parse <u>H</u>essian <u>a</u>utomatic <u>d</u>ifferentiation <u>o</u>ptimization frame<u>w</u>ork.

Author and maintainer: Jean-Philippe Rasigade,
University of Lyon, <jean-philippe.rasigade@univ-lyon1.fr><br/>
Rshadow is licensed under CC BY - NC - ND 4.0, https://creativecommons.org/licenses/by-nc-nd/4.0/<br/>
**NON-COMMERCIAL USE ONLY.** Alternative private/commercial licensing options available upon request to the author.

## Installation and requirements

`Rshadow` requires packages `RcppEigen` and `devtools` (for GitHub installation)

Install using `devtools::install_github("rasigadelab/Rshadow")`

Be patient (like, prepare coffee): C++ compilation may take several minutes.

## Rshadow Hello World

`Rshadow` is an *operator overloading* package, meaning that it exposes objects that behave as regular R numeric vectors, matrices or arrays, but that exhibit special functionality under the hood. These objects are the *tape*, which records instructions, and the *spies*, which mimic basic R objects, spy the functions that we apply on them, and record the functions on the tape to let the underlying C++ framework handle the optimization tasks using its own compiler.

The basic steps of optimization are:
1. Declare an objective function using `tape_new`
2. Declare parameters of interest using `spy`
3. Write down the objective function with the usual R syntax
4. Call `maximize` (Rshadow is targeted at maximum likelihood estimation)
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
