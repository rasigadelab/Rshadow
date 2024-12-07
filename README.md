# Rshadow

This repository holds pre-compiled binaries for the `Rshadow` automatic optimizer.

(c)2022-2024 Jean-Philippe Rasigade, <jean-philippe.rasigade@univ-lyon1.fr>

The package source code is currently under embargo but will become eventually open-source. Feel free to use the pre-compiled binaries for any non-commercial purpose. 

*Commercial use is currently not permitted. Sorry about that.*

## What does it do?

`Rshadow` is an automatic optimizer for R. It is a high-performance second-order optimizer with automatic differentiation and operator overloading. It exposes the SHADOw library to the R frontend. SHADOw stands for Sparse Hessian AutoDiff Optimization frameWork.

## Install (Windows only)

```R
install.packages("https://github.com/rasigadelab/Rshadow/raw/main/Rshadow_0.1.zip", repos = NULL)
```
