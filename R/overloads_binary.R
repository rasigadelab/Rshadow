#/***********************************************************************************/
#/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
#/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
#/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
#/* Alternative private/commercial licensing available upon request                 */
#/***********************************************************************************/

### OPERATORS ##############################################

# /*** PLUS ******************************************/

#' @export
setMethod("+", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_plus_Tensor_Tensor(e1, e2))
#' @export
setMethod("+", c(e1="shadow_tensor", e2="shadow_spy"), 
          function(e1, e2) .shadow_plus_Tensor_Spy(e1, e2))
#' @export
setMethod("+", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_plus_Spy_Tensor(e1, e2))
#' @export
setMethod("+", c(e1="shadow_spy", e2="shadow_spy"), 
          function(e1, e2) .shadow_plus_Spy_Spy(e1, e2))
#' @export
setMethod("+", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_plus_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod("+", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_plus_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod("+", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_plus_Spy_Tensor(e1, tensor(e2)))
#' @export
setMethod("+", c(e1="numeric", e2="shadow_spy"), 
          function(e1, e2) .shadow_plus_Tensor_Spy(tensor(e1), e2))

# /*** MINUS ******************************************/

#' @export
setMethod("-", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_minus_Tensor_Tensor(e1, e2))
#' @export
setMethod("-", c(e1="shadow_tensor", e2="shadow_spy"), 
          function(e1, e2) .shadow_minus_Tensor_Spy(e1, e2))
#' @export
setMethod("-", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_minus_Spy_Tensor(e1, e2))
#' @export
setMethod("-", c(e1="shadow_spy", e2="shadow_spy"), 
          function(e1, e2) .shadow_minus_Spy_Spy(e1, e2))
#' @export
setMethod("-", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_minus_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod("-", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_minus_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod("-", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_minus_Spy_Tensor(e1, tensor(e2)))
#' @export
setMethod("-", c(e1="numeric", e2="shadow_spy"), 
          function(e1, e2) .shadow_minus_Tensor_Spy(tensor(e1), e2))

# /*** MULTIPLIES ******************************************/

#' @export
setMethod("*", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_multiplies_Tensor_Tensor(e1, e2))
#' @export
setMethod("*", c(e1="shadow_tensor", e2="shadow_spy"), 
          function(e1, e2) .shadow_multiplies_Tensor_Spy(e1, e2))
#' @export
setMethod("*", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_multiplies_Spy_Tensor(e1, e2))
#' @export
setMethod("*", c(e1="shadow_spy", e2="shadow_spy"), 
          function(e1, e2) .shadow_multiplies_Spy_Spy(e1, e2))
#' @export
setMethod("*", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_multiplies_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod("*", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_multiplies_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod("*", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_multiplies_Spy_Tensor(e1, tensor(e2)))
#' @export
setMethod("*", c(e1="numeric", e2="shadow_spy"), 
          function(e1, e2) .shadow_multiplies_Tensor_Spy(tensor(e1), e2))

# /*** DIVIDES ******************************************/

#' @export
setMethod("/", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_divides_Tensor_Tensor(e1, e2))
#' @export
setMethod("/", c(e1="shadow_tensor", e2="shadow_spy"), 
          function(e1, e2) .shadow_divides_Tensor_Spy(e1, e2))
#' @export
setMethod("/", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_divides_Spy_Tensor(e1, e2))
#' @export
setMethod("/", c(e1="shadow_spy", e2="shadow_spy"), 
          function(e1, e2) .shadow_divides_Spy_Spy(e1, e2))
#' @export
setMethod("/", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_divides_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod("/", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_divides_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod("/", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_divides_Spy_Tensor(e1, tensor(e2)))
#' @export
setMethod("/", c(e1="numeric", e2="shadow_spy"), 
          function(e1, e2) .shadow_divides_Tensor_Spy(tensor(e1), e2))

# /*** POWER ******************************************/

#' @export
setMethod("^", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_pow_Tensor_Tensor(e1, e2))
#' @export
setMethod("^", c(e1="shadow_tensor", e2="shadow_spy"), 
          function(e1, e2) .shadow_pow_Tensor_Spy(e1, e2))
#' @export
setMethod("^", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_pow_Spy_Tensor(e1, e2))
#' @export
setMethod("^", c(e1="shadow_spy", e2="shadow_spy"), 
          function(e1, e2) .shadow_pow_Spy_Spy(e1, e2))
#' @export
setMethod("^", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_pow_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod("^", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_pow_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod("^", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_pow_Spy_Tensor(e1, tensor(e2)))
#' @export
setMethod("^", c(e1="numeric", e2="shadow_spy"), 
          function(e1, e2) .shadow_pow_Tensor_Spy(tensor(e1), e2))

# /*** LESS THAN ******************************************/

#' @export
setMethod("<", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_less_Tensor_Tensor(e1, e2))
#' @export
setMethod("<", c(e1="shadow_tensor", e2="shadow_spy"), 
          function(e1, e2) .shadow_less_Tensor_Spy(e1, e2))
#' @export
setMethod("<", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_less_Spy_Tensor(e1, e2))
#' @export
setMethod("<", c(e1="shadow_spy", e2="shadow_spy"), 
          function(e1, e2) .shadow_less_Spy_Spy(e1, e2))
#' @export
setMethod("<", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_less_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod("<", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_less_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod("<", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_less_Spy_Tensor(e1, tensor(e2)))
#' @export
setMethod("<", c(e1="numeric", e2="shadow_spy"), 
          function(e1, e2) .shadow_less_Tensor_Spy(tensor(e1), e2))

# /*** LESS THAN OR EQUAL *************************************/

#' @export
setMethod("<=", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_less_equal_Tensor_Tensor(e1, e2))
#' @export
setMethod("<=", c(e1="shadow_tensor", e2="shadow_spy"), 
          function(e1, e2) .shadow_less_equal_Tensor_Spy(e1, e2))
#' @export
setMethod("<=", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_less_equal_Spy_Tensor(e1, e2))
#' @export
setMethod("<=", c(e1="shadow_spy", e2="shadow_spy"), 
          function(e1, e2) .shadow_less_equal_Spy_Spy(e1, e2))
#' @export
setMethod("<=", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_less_equal_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod("<=", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_less_equal_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod("<=", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_less_equal_Spy_Tensor(e1, tensor(e2)))
#' @export
setMethod("<=", c(e1="numeric", e2="shadow_spy"), 
          function(e1, e2) .shadow_less_equal_Tensor_Spy(tensor(e1), e2))

# /*** GREATER THAN ******************************************/

#' @export
setMethod(">", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_greater_Tensor_Tensor(e1, e2))
#' @export
setMethod(">", c(e1="shadow_tensor", e2="shadow_spy"), 
          function(e1, e2) .shadow_greater_Tensor_Spy(e1, e2))
#' @export
setMethod(">", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_greater_Spy_Tensor(e1, e2))
#' @export
setMethod(">", c(e1="shadow_spy", e2="shadow_spy"), 
          function(e1, e2) .shadow_greater_Spy_Spy(e1, e2))
#' @export
setMethod(">", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_greater_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod(">", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_greater_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod(">", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_greater_Spy_Tensor(e1, tensor(e2)))
#' @export
setMethod(">", c(e1="numeric", e2="shadow_spy"), 
          function(e1, e2) .shadow_greater_Tensor_Spy(tensor(e1), e2))

# /*** GREATER THAN OR EQUAL *************************************/

#' @export
setMethod(">=", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_greater_equal_Tensor_Tensor(e1, e2))
#' @export
setMethod(">=", c(e1="shadow_tensor", e2="shadow_spy"), 
          function(e1, e2) .shadow_greater_equal_Tensor_Spy(e1, e2))
#' @export
setMethod(">=", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_greater_equal_Spy_Tensor(e1, e2))
#' @export
setMethod(">=", c(e1="shadow_spy", e2="shadow_spy"), 
          function(e1, e2) .shadow_greater_equal_Spy_Spy(e1, e2))
#' @export
setMethod(">=", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_greater_equal_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod(">=", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_greater_equal_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod(">=", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_greater_equal_Spy_Tensor(e1, tensor(e2)))
#' @export
setMethod(">=", c(e1="numeric", e2="shadow_spy"), 
          function(e1, e2) .shadow_greater_equal_Tensor_Spy(tensor(e1), e2))

# /*** DOT PRODUCT ******************************************/

setGeneric("dot", function(e1, e2) standardGeneric("dot") )

#' @export
setMethod("dot", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_dot_Tensor_Tensor(e1, e2))
#' @export
setMethod("dot", c(e1="shadow_tensor", e2="shadow_spy"), 
          function(e1, e2) .shadow_dot_Tensor_Spy(e1, e2))
#' @export
setMethod("dot", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_dot_Spy_Tensor(e1, e2))
#' @export
setMethod("dot", c(e1="shadow_spy", e2="shadow_spy"), 
          function(e1, e2) .shadow_dot_Spy_Spy(e1, e2))
#' @export
setMethod("dot", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_dot_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod("dot", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_dot_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod("dot", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_dot_Spy_Tensor(e1, tensor(e2)))
#' @export
setMethod("dot", c(e1="numeric", e2="shadow_spy"), 
          function(e1, e2) .shadow_dot_Tensor_Spy(tensor(e1), e2))


# /*** MATRIX PRODUCT ******************************************/


#' @export
setMethod("%*%", c(x="shadow_tensor", y="shadow_tensor"), 
          function(x, y) .shadow_matmult_Tensor_Tensor(x, y))
#' @export
setMethod("%*%", c(x="shadow_tensor", y="shadow_spy"), 
          function(x, y) .shadow_matmult_Tensor_Spy(x, y))
#' @export
setMethod("%*%", c(x="shadow_spy", y="shadow_tensor"), 
          function(x, y) .shadow_matmult_Spy_Tensor(x, y))
#' @export
setMethod("%*%", c(x="shadow_spy", y="shadow_spy"), 
          function(x, y) .shadow_matmult_Spy_Spy(x, y))
#' @export
setMethod("%*%", c(x="shadow_tensor", y="array"), 
          function(x, y) .shadow_matmult_Tensor_Tensor(x, tensor(y)))
#' @export
setMethod("%*%", c(x="array", y="shadow_tensor"), 
          function(x, y) .shadow_matmult_Tensor_Tensor(tensor(x), y))
#' @export
setMethod("%*%", c(x="shadow_spy", y="array"), 
          function(x, y) .shadow_matmult_Spy_Tensor(x, tensor(y)))
#' @export
setMethod("%*%", c(x="array", y="shadow_spy"), 
          function(x, y) .shadow_matmult_Tensor_Spy(tensor(x), y))


# /*** SUM LOG BERN ******************************************/

# Custom operator for logistic regression:
# sumlogbern(x, b) = sum b log x + (1 - b) log 1 - x
# this is the model log likelihood of observed events b \in {0, 1}
# each with probability x

setGeneric("sumlogbern", function(e1, e2) standardGeneric("sumlogbern") )

#' @export
setMethod("sumlogbern", c(e1="shadow_tensor", e2="shadow_tensor"), 
          function(e1, e2) .shadow_sumlogbern_Tensor_Tensor(e1, e2))
#' @export
setMethod("sumlogbern", c(e1="shadow_tensor", e2="numeric"), 
          function(e1, e2) .shadow_sumlogbern_Tensor_Tensor(e1, tensor(e2)))
#' @export
setMethod("sumlogbern", c(e1="numeric", e2="shadow_tensor"), 
          function(e1, e2) .shadow_sumlogbern_Tensor_Tensor(tensor(e1), e2))
#' @export
setMethod("sumlogbern", c(e1="shadow_spy", e2="shadow_tensor"), 
          function(e1, e2) .shadow_sumlogbern_Spy_Tensor(e1, e2))
#' @export
setMethod("sumlogbern", c(e1="shadow_spy", e2="numeric"), 
          function(e1, e2) .shadow_sumlogbern_Spy_Tensor(e1, tensor(e2)))