#/****************************************************************************/
#/* SHADOw framework, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>      */
#/* Licensed under the GNU Affero GPL Version 3                              */
#/****************************************************************************/

library(Rshadow)
library(testthat)

test_that("spy is subsettable with [] operator", {
  tape <- tape_new()
  x_ <- tensor(matrix(rnorm(16), 4, 4))
  x  <- spy(x_, tape)
  
  expect_equal(as.numeric(x_)[1], as.numeric(x[1]))
  expect_equal(as.numeric(x_)[2,3], as.numeric(x[2,3]))
  expect_error(x[17])
  expect_error(x[5,1])
  expect_error(x[1,5])
  expect_error(x[1:2])
  expect_error(x[,1])
  expect_error(x[1,])
})