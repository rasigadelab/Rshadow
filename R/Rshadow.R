#/***********************************************************************************/
#/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
#/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
#/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
#/* Alternative private/commercial licensing available upon request                 */
#/***********************************************************************************/

#' @export
rebuild <- function() {
  pkgload:::unload("Rshadow")
  Rcpp:::compileAttributes()
  roxygen2::roxygenize(roclets = c("collate", "rd"), clean = TRUE)
  ret <- system(
    "Rcmd INSTALL . --no-multiarch --with-keep.source"
    , intern = FALSE)
  library(Rshadow)
}

### COMMON S4 CLASS DEFINITIONS #######################################
# For some reason, S4 classes must be defined in the Rshadow.R
# file. If they're moved to their own file, lots of 'no defined class'
# warnings emerge, even if tests run fine.

# S4 classes
# https://teuder.github.io/rcpp4everyone_en/160_s3_s4.html

#' @exportClass shadow_tape
setClass("shadow_tape", representation(ptr = "externalptr"))
#' @exportClass shadow_tensor
setClass("shadow_tensor", representation(ptr = "externalptr"))
#' @exportClass shadow_spy
setClass("shadow_spy", representation(ptr = "externalptr"))
#' @exportClass shadow_trace
setClass("shadow_trace", representation(ptr = "externalptr"))
#' @exportClass shadow_solver
setClass("shadow_solver", representation(ptr = "externalptr"))

### S4 TAPE = shadow_tape ###############################################

#' @export
tape_new <- function() .shadow_tape_new()

#' @exportClass shadow_tape_summary
setClass("shadow_tape_summary", representation(
    input_size = "integer", 
    trace_size = "integer")
)

### S4 SOLVER = shadow_solver ##########################################

#' @exportClass shadow_solver_config
setClass("shadow_solver_config",
  representation(
    max_iterations = "integer",
    objective_tolerance = "numeric",
    diagnostic_mode = "logical",
    max_regularization_attempts = "integer",
    regularization_damping_factor = "numeric",
    brent_tolerance_factor = "numeric",
    brent_boundary_left = "numeric",
    brent_boundary_right = "numeric",
    brent_feasible_search_restriction_factor = "numeric"
  ),
  prototype = list(
    max_iterations = 1000L,
    objective_tolerance = 1e-3,
    diagnostic_mode = FALSE,
    max_regularization_attempts = 10L,
    regularization_damping_factor = 2.0,
    brent_tolerance_factor = 1.0,
    brent_boundary_left = -1.0,
    brent_boundary_right = +2.0,
    brent_feasible_search_restriction_factor = 0.75
  )
)

#' @exportClass  shadow_solver_state
setClass("shadow_solver_state", 
  representation(
    iter = "integer",
    objective_initial = "numeric",
    objective_final = "numeric",
    lambda = "numeric",
    parameters = "numeric",
    gradient = "numeric",
    hessian = "numeric",
    direction = "numeric",
    brent_left = "numeric",
    brent_right = "numeric",
    optstep = "numeric",
    n_eval = "integer",
    n_solves = "integer",
    n_regul = "integer",
    using_gradient_fallback = "logical"
  )
)

#' @export
setGeneric("solver",    function(trace, config) standardGeneric("solver") )
setMethod("solver",
    c(trace = "shadow_trace", config = "shadow_solver_config"),
    function(trace, config) {
        .shadow_get_solver_with_config(trace, config)
    }
)
setMethod("solver",
    c(trace = "shadow_trace", config = "missing"),
    function(trace, config) {
        config <- shadow_solver_config_new()
        .shadow_get_solver_with_config(trace, config)
    }
)

### METHODS ##############################################





#' @export
setMethod("summary", "shadow_tape", function(object) shadow_tape_summary(object))


######### SHORTHANDS

setGeneric("maximize",    function(tape, config) standardGeneric("maximize") )
setMethod("maximize",
  c(tape = "shadow_tape", config = "shadow_solver_config"),
  function(tape, config) {
    trace  <- shadow_trace_new(tape)
    solver <- solver(trace, config)
    solver <- shadow_solver_maximize(solver)
    attr(trace, "solver") <- solver
    return(trace)
  }
)
setMethod("maximize",
  c(tape = "shadow_tape", config = "missing"),
  function(tape, config) {
    trace  <- shadow_trace_new(tape)
    solver <- solver(trace)
    solver <- shadow_solver_maximize(solver)
    attr(trace, "solver") <- solver
    return(trace)
  }
)

setGeneric("hessian", function(trace) standardGeneric("hessian") )
setMethod("hessian", "shadow_trace", function(trace) .shadow_trace_hessian(trace))

