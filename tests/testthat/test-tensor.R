#/****************************************************************************/
#/* SHADOw framework, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>      */
#/* Licensed under the GNU Affero GPL Version 3                              */
#/****************************************************************************/

library(Rshadow)
library(testthat)

test_that("tensor accepts scalar, vector, matrix and array types", {
  expect_s4_class(tensor(2.3), "shadow_tensor")
  expect_s4_class(tensor(rnorm(10)), "shadow_tensor")
  expect_s4_class(tensor(matrix(rnorm(16), 4, 4)), "shadow_tensor")
  expect_s4_class(tensor(array(rnorm(5*4*3*2), c(5,4,3,2))), "shadow_tensor")
  
  # Tensor dimension
  expect_equal(dim(tensor(matrix(rnorm(16), 4, 4))), c(4, 4))
  expect_equal(dim(tensor(array(rnorm(5*4*3*2), c(5,4,3,2)))), c(5,4,3,2))
})


test_that("tensor overloads for binary operators match the results of R operators", {
  # Tensor binary operators
  a_ <- pi
  b_ <- exp(pi)
  a <- tensor(a_)
  b <- tensor(b_)

  expect_equal(a_ + b_, as.numeric(a + b))
  expect_equal(a_ + b_, as.numeric(a + b_))
  expect_equal(a_ + b_, as.numeric(a_ + b))
  
  expect_equal(a_ - b_, as.numeric(a - b))
  expect_equal(a_ - b_, as.numeric(a - b_))
  expect_equal(a_ - b_, as.numeric(a_ - b))
  
  expect_equal(a_ * b_, as.numeric(a * b))
  expect_equal(a_ * b_, as.numeric(a * b_))
  expect_equal(a_ * b_, as.numeric(a_ * b))
  
  expect_equal(a_ / b_, as.numeric(a / b))
  expect_equal(a_ / b_, as.numeric(a / b_))
  expect_equal(a_ / b_, as.numeric(a_ / b))
  
  expect_equal(a_ ^ b_, as.numeric(a ^ b))
  expect_equal(a_ ^ b_, as.numeric(a ^ b_))
  expect_equal(a_ ^ b_, as.numeric(a_ ^ b))
})

test_that("tensor overloads for comparison operators match the results of R operators", {
  # Tensor binary operators
  a_ <- pi
  b_ <- exp(pi)
  a <- tensor(a_)
  b <- tensor(b_)
  
  expect_equal(as.numeric(a_ < b_), as.numeric(a  < b))
  expect_equal(as.numeric(a_ < b_), as.numeric(a  < b_))
  expect_equal(as.numeric(a_ < b_), as.numeric(a_ < b))
  expect_equal(as.numeric(a_ < a_), as.numeric(a  < a))
  expect_equal(as.numeric(a_ < a_), as.numeric(a  < a_))
  expect_equal(as.numeric(a_ < a_), as.numeric(a_ < a))

  expect_equal(as.numeric(a_ <= b_), as.numeric(a  <= b))
  expect_equal(as.numeric(a_ <= b_), as.numeric(a  <= b_))
  expect_equal(as.numeric(a_ <= b_), as.numeric(a_ <= b))
  expect_equal(as.numeric(a_ <= a_), as.numeric(a  <= a))
  expect_equal(as.numeric(a_ <= a_), as.numeric(a  <= a_))
  expect_equal(as.numeric(a_ <= a_), as.numeric(a_ <= a))

  expect_equal(as.numeric(a_ > b_), as.numeric(a  > b))
  expect_equal(as.numeric(a_ > b_), as.numeric(a  > b_))
  expect_equal(as.numeric(a_ > b_), as.numeric(a_ > b))
  expect_equal(as.numeric(a_ > a_), as.numeric(a  > a))
  expect_equal(as.numeric(a_ > a_), as.numeric(a  > a_))
  expect_equal(as.numeric(a_ > a_), as.numeric(a_ > a))

  expect_equal(as.numeric(a_ >= b_), as.numeric(a  >= b))
  expect_equal(as.numeric(a_ >= b_), as.numeric(a  >= b_))
  expect_equal(as.numeric(a_ >= b_), as.numeric(a_ >= b))
  expect_equal(as.numeric(a_ >= a_), as.numeric(a  >= a))
  expect_equal(as.numeric(a_ >= a_), as.numeric(a  >= a_))
  expect_equal(as.numeric(a_ >= a_), as.numeric(a_ >= a))
})

test_that("tensor overloads for unary functions match the results of R functions", {
  # Tensor unary operators  
  a_ <- pi
  a <- tensor(a_)
  
  expect_equal(-a_, as.numeric(-a))
  expect_equal(log(a_), as.numeric(log(a)))
  expect_equal(log1p(a_), as.numeric(log1p(a)))
  expect_equal(log1m(-a_), as.numeric(log1m(-a)))
  expect_equal(exp(a_), as.numeric(exp(a)))
  expect_equal(lgamma(a_), as.numeric(lgamma(a)))
  expect_equal(logit(1/a_), as.numeric(logit(1/a)))
  expect_equal(logistic(a_), as.numeric(logistic(a)))
})

test_that("tensor aggregators match the results of R functions", {
  x_ <- rnorm(10)
  y_ <- rnorm(10)
  x  <- tensor(x_)
  y  <- tensor(y_)
  expect_equal(sum(x_), as.numeric(sum(x)))
  expect_equal(sum(x_^2), as.numeric(sumsq(x)))
  
  expect_equal(as.double(t(x_) %*% y_), as.numeric(dot(x,  y )))
  expect_equal(as.double(t(x_) %*% y_), as.numeric(dot(x_, y )))
  expect_equal(as.double(t(x_) %*% y_), as.numeric(dot(x,  y_)))
  
  # Custom operator for logistic regression:
  # sumlogbern(p, b) = sum b log p + (1 - b) log 1 - p
  p_ <- runif(10)    
  b_ <- sample(c(0,1), 10, replace = TRUE)
  p  <- tensor(p_)
  b  <- tensor(b_)
  expect_equal(sum( b_ * log(p_) + (1 - b_) * log1p(-p_) ), as.numeric(sumlogbern(p, b)))
})


test_that("matrix multiplication preserves dimensions", {
  x_ <- matrix(rnorm(12*3), 12, 3)
  y_ <- matrix(rnorm(3*7), 3, 7)
  x  <- tensor(x_)
  y  <- tensor(y_)
  expect_equal(x_ %*% y_, as.numeric(x  %*% y))
  expect_equal(x_ %*% y_, as.numeric(x  %*% y_))
  expect_equal(x_ %*% y_, as.numeric(x_ %*% y))
})

test_that("tensor is subsettable with [] operator", {
  x_ <- tensor(matrix(rnorm(16), 4, 4))
  
  expect_equal(as.numeric(x_)[1], as.numeric(x_[1]))
  expect_equal(as.numeric(x_)[1,], as.numeric(x_[1,]))
  expect_equal(as.numeric(x_)[,1], as.numeric(x_[,1]))
  
})