#/****************************************************************************/
#/* SHADOw framework, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>      */
#/* Licensed under the GNU Affero GPL Version 3                              */
#/****************************************************************************/

library(Rshadow)
library(testthat)

test_that("solver finds the maximum of -x²", {
    tape <- tape_new()
    x <- spy(1.5, tape) # Starting point
    y <- -x^2

    trace <- maximize(tape)

    xx <- read(x, trace)
  
    expect_equal(as.numeric(xx), 0)
})

test_that("solver finds the mean and standard deviation of normal deviates", {
  # Data
  set.seed(1)
  n <- 100
  x <- rnorm(n)
  
  tape <- tape_new()
  
  # Inputs
  mu <- spy(0, tape)
  sigma_log <- spy(0, tape)
  
  # Problem
  sigma <- exp(sigma_log)
  ll <- sum(ll_norm(x, mu, sigma))
  
  # Result
  trace <- maximize(tape)
  
  mu_opt <- read(mu, trace)
  sigma_opt <- read(sigma, trace)
  
  expect_lt(squared_diff(mu_opt, mean(x)), 1e-12)
  expect_lt(squared_diff(sigma_opt, sd(x) / sqrt(n/(n-1))), 1e-12)
})

test_that("solver in diagnostic mode stores step information", {
  
  # Data
  set.seed(1)
  n <- 100
  x <- rnorm(n)
  
  tape <- tape_new()
  
  # Inputs
  mu <- spy(0, tape)
  sigma_log <- spy(0, tape)
  
  # Problem
  sigma <- exp(sigma_log)
  ll <- sum(ll_norm(x, mu, sigma))
  
  config <- shadow_solver_config_new()
  config@diagnostic_mode = TRUE
  trace <- maximize(tape, config)
  
  states <- shadow_extract_solver_states(attr(trace, "solver"))
  
  expect_gt(length(states), 0)
})