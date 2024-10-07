#/***********************************************************************************/
#/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
#/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
#/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
#/* Alternative private/commercial licensing available upon request                 */
#/***********************************************************************************/

### UNARY FUNCTIONS ##############################################

# Create generic method for Rshadow-specific function names
setGeneric("log1m",    function(x) standardGeneric("log1m") )
setGeneric("logit",    function(x) standardGeneric("logit") )
setGeneric("logistic", function(x) standardGeneric("logistic") )
setGeneric("sumsq",    function(x) standardGeneric("sumsq") )

#' @export
setMethod("-", signature(e1 = "shadow_tensor", e2 = "missing"), function(e1) .shadow_negate_Tensor(e1))
#' @export
setMethod("-", signature(e1 = "shadow_spy", e2 = "missing"), function(e1) .shadow_negate_Spy(e1))
#' @export
setMethod("log", signature(x = "shadow_tensor"), function(x) .shadow_log_Tensor(x))
#' @export
setMethod("log", signature(x = "shadow_spy"), function(x) .shadow_log_Spy(x))
#' @export
setMethod("log1p", signature(x = "shadow_tensor"), function(x) .shadow_log1p_Tensor(x))
#' @export
setMethod("log1p", signature(x = "shadow_spy"), function(x) .shadow_log1p_Spy(x))
#' @export
setMethod("log1m", signature(x = "numeric"), function(x) log1p(-x))
#' @export
setMethod("log1m", signature(x = "shadow_tensor"), function(x) .shadow_log1m_Tensor(x))
#' @export
setMethod("log1m", signature(x = "shadow_spy"), function(x) .shadow_log1m_Spy(x))
#' @export
setMethod("exp", signature(x = "shadow_tensor"), function(x) .shadow_exp_Tensor(x))
#' @export
setMethod("exp", signature(x = "shadow_spy"), function(x) .shadow_exp_Spy(x))
#' @export
setMethod("lgamma", signature(x = "shadow_tensor"), function(x) .shadow_lgamma_Tensor(x))
#' @export
setMethod("lgamma", signature(x = "shadow_spy"), function(x) .shadow_lgamma_Spy(x))
#' @export
setMethod("logit", signature(x = "numeric"), function(x) log(x) - log1p(-x))
#' @export
setMethod("logit", signature(x = "shadow_tensor"), function(x) .shadow_logit_Tensor(x))
#' @export
setMethod("logit", signature(x = "shadow_spy"), function(x) .shadow_logit_Spy(x))
#' @export
setMethod("logistic", signature(x = "numeric"), function(x) 1/(1+exp(-x)))
#' @export
setMethod("logistic", signature(x = "shadow_tensor"), function(x) .shadow_logistic_Tensor(x))
#' @export
setMethod("logistic", signature(x = "shadow_spy"), function(x) .shadow_logistic_Spy(x))
#' @export
setMethod("sum", signature(x = "shadow_tensor"), function(x) .shadow_sum_Tensor(x))
#' @export
setMethod("sum", signature(x = "shadow_spy"), function(x) .shadow_sum_Spy(x))
#' @export
setMethod("sumsq", signature(x = "shadow_tensor"), function(x) .shadow_sumsq_Tensor(x))
#' @export
setMethod("sumsq", signature(x = "shadow_spy"), function(x) .shadow_sumsq_Spy(x))