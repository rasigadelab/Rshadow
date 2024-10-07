/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/

/* This file contains the R-facing functions to manipulate
tensors, spies, tapes, etc. 

REMARK. All these methods are packed together in one file
because R's GCC is slooooow and not parallel. Making more
files just increases the compilation time. */

#include "s4_io.h"
#include <convert.h>

/****************************************************/
/*** TENSOR *****************************************/
/****************************************************/

// [[Rcpp::export(.shadow_tensor_new)]]
Rcpp::S4 shadow_tensor_new( Rcpp::NumericVector x ) {
  
  Tensor* tensor_ptr = nullptr;
  
  /* Manage dimension */
  if(x.hasAttribute("dim")) {
    Rcpp::IntegerVector dim = x.attr("dim");
    std::vector< size_t > cdim(dim.size());
    for(int i = 0; i < dim.length(); i++) {
      cdim[i] = size_t(dim[i]);
    }
    tensor_ptr = new Tensor(cdim);

  }
  else {
    /* .length() returns int so we must wrap it */
    tensor_ptr = new Tensor( TensorDim( { size_t( x.length() ) } ));
  }
  
  /* Set values, don't entangle input and Tensor */
  for(size_t i = 0; i < tensor_ptr->size(); i++) {
    tensor_ptr->val[i] = x[i];
  }
  return wrap_ptr_S4(tensor_ptr);
}

// [[Rcpp::export(.shadow_tensor_as_numeric)]]
Rcpp::NumericVector shadow_tensor_as_numeric(Rcpp::S4 shadow_tensor) {
  Tensor& tensor = from_S4<Tensor>(shadow_tensor);
  return NumericVector_from_Tensor(tensor);
}

/****************************************************/
/*** SPY ********************************************/
/****************************************************/

// [[Rcpp::export(.shadow_spy_new)]]
Rcpp::S4 shadow_spy_new(Rcpp::S4 tensor_in, Rcpp::S4 tape_in) {
    Tensor& tensor = from_S4<Tensor>(tensor_in);
    Tape& tape = from_S4<Tape>(tape_in);
    Spy spy(tensor, tape);
    return to_S4(spy);
}

// [[Rcpp::export(.shadow_spy_as_numeric)]]
Rcpp::NumericVector shadow_spy_as_numeric(Rcpp::S4 shadow_spy) {
    Spy& spy = from_S4<Spy>(shadow_spy);
    return NumericVector_from_Tensor(spy.tensor());
}

// [[Rcpp::export(.shadow_spy_get_trace_index)]]
Rcpp::IntegerVector shadow_spy_get_trace_index(Rcpp::S4 shadow_spy) {
    Spy& spy = from_S4<Spy>(shadow_spy);
    return Rcpp::wrap(spy.tape_begin());
}

/* Transform a list of contiguous scalar spies into a single spy */

// [[Rcpp::export]]
Rcpp::S4 shadow_spy_bind_list(Rcpp::List shadow_spy_list) {

    Spy& spy_begin = from_S4<Spy>(shadow_spy_list[0]);
    index_t index_begin = spy_begin.tape_begin();
    Tape& tape = spy_begin.tape;
    std::vector< double > x(shadow_spy_list.size());

    for(int i = 0; i < shadow_spy_list.size(); i++) {
      Spy& spy = from_S4<Spy>(shadow_spy_list[i]);
      if(spy.is_scalar() == false) throw std::logic_error("Non-scalar Spy not allowed");
      if(spy.tape_begin() != index_begin + i) throw std::logic_error("Non-contiguous spies not allowed");
      x[i] = spy.tensor().scalar();
    }    
    Spy spy(x, tape, index_begin);
    return to_S4(spy);
}

// [[Rcpp::export(.shadow_spy_subset_i)]]
Rcpp::S4 shadow_spy_subset_i(Rcpp::S4 spy_in, index_t i) {
    Spy& spy = from_S4<Spy>(spy_in);
    if (i - 1 >= index_t(spy.size())) throw std::out_of_range("i out of range");
    Spy spy_i = spy[i - 1];
    return to_S4(spy_i);
}

// [[Rcpp::export(.shadow_spy_subset_i_j)]]
Rcpp::S4 shadow_spy_subset_i_j(Rcpp::S4 spy_in, index_t i, index_t j) {
    Spy& spy = from_S4<Spy>(spy_in);
    if (spy.dim.size() != 2) throw std::logic_error("Spy is not a matrix");
    if (i - 1 >= index_t(spy.dim[0])) throw std::out_of_range("i out of range");
    if (j - 1 >= index_t(spy.dim[1])) throw std::out_of_range("j out of range");
    Spy spy_ij = spy(i - 1, j - 1);
    return to_S4(spy_ij);
}

// [[Rcpp::export(.shadow_spy_subset_i_j_k)]]
Rcpp::S4 shadow_spy_subset_i_j_k(Rcpp::S4 spy_in, index_t i, index_t j, index_t k) {
    Spy& spy = from_S4<Spy>(spy_in);
    if (spy.dim.size() != 2) throw std::logic_error("Spy is not a cube");
    if (i - 1 >= index_t(spy.dim[0])) throw std::out_of_range("i out of range");
    if (j - 1 >= index_t(spy.dim[1])) throw std::out_of_range("j out of range");
    if (k - 1 >= index_t(spy.dim[2])) throw std::out_of_range("k out of range");
    Spy spy_ijk = spy(i - 1, j - 1, k - 1);
    return to_S4(spy_ijk);
}

// [[Rcpp::export(.shadow_spy_read_on_trace)]]
Rcpp::NumericVector shadow_spy_read_on_trace(Rcpp::S4 spy_in, Rcpp::S4 trace_in) {
    Trace& trace = from_S4<Trace>(trace_in);
    Spy& spy = from_S4<Spy>(spy_in);
    Tensor tensor(spy.dim);
    std::copy(trace.values.begin() + spy.tape_begin(), trace.values.begin() + spy.tape_end(), tensor.val.begin());
    return NumericVector_from_Tensor(tensor);
}

/****************************************************/
/*** TAPE *******************************************/
/****************************************************/

// [[Rcpp::export(.shadow_tape_new)]]
Rcpp::S4 shadow_tape_new() {
    Tape tape;
    return to_S4(tape);
}

// [[Rcpp::export]]
Rcpp::S4 shadow_tape_summary(Rcpp::S4 tape_in) {
  Tape& tape = from_S4<Tape>(tape_in);

  Rcpp::S4 result("shadow_tape_summary");
  result.slot("input_size") = Rcpp::wrap(tape.input_size());
  result.slot("trace_size") = Rcpp::wrap(tape.trace_size());
  return result;
}

/****************************************************/
/*** TRACE ******************************************/
/****************************************************/

// [[Rcpp::export]]
Rcpp::S4 shadow_trace_new(Rcpp::S4 tape_in) {
  Tape& tape = from_S4<Tape>(tape_in);
  Trace trace = Trace(tape).play();
  return to_S4(trace);
}

// [[Rcpp::export]]
Rcpp::NumericVector shadow_trace_values(Rcpp::S4 trace_in) {
  Trace& trace = from_S4<Trace>(trace_in);
  return Rcpp::wrap(trace.values);
}

// [[Rcpp::export]]
Rcpp::NumericVector shadow_trace_adjoints(Rcpp::S4 trace_in) {
  Trace& trace = from_S4<Trace>(trace_in);
  return Rcpp::wrap(trace.adjoints);
}

// [[Rcpp::export(.shadow_trace_hessian)]]
Rcpp::NumericVector shadow_trace_hessian(Rcpp::S4 trace_in) {
  Trace& trace = from_S4<Trace>(trace_in);
  Tensor hessian_tensor = Convert::Tensor_from_SparseSymMat(trace.hessian);
  return NumericVector_from_Tensor(hessian_tensor);
}

/****************************************************/
/*** SOLVER *****************************************/
/****************************************************/

// [[Rcpp::export(.shadow_solver_new)]]
Rcpp::S4 shadow_solver_new(Rcpp::S4 trace_in) {
  Trace& trace = from_S4<Trace>(trace_in);
  Solver solver(trace);
  return to_S4(solver);
}

#define FROM_S4_CONFIG_SLOT(slot_name) \
config.slot_name = config_in.slot(#slot_name);

/* Makes a SolverConfig objet from the corresponding S4 class */
SolverConfig shadow_make_solver_config(Rcpp::S4 config_in) {
  SolverConfig config;
  FROM_S4_CONFIG_SLOT(max_iterations)
  FROM_S4_CONFIG_SLOT(objective_tolerance)
  FROM_S4_CONFIG_SLOT(diagnostic_mode)
  FROM_S4_CONFIG_SLOT(max_regularization_attempts)
  FROM_S4_CONFIG_SLOT(regularization_damping_factor)
  FROM_S4_CONFIG_SLOT(brent_tolerance_factor)
  FROM_S4_CONFIG_SLOT(brent_boundary_left)
  FROM_S4_CONFIG_SLOT(brent_boundary_right)
  FROM_S4_CONFIG_SLOT(brent_feasible_search_restriction_factor)
  return config;
}
#undef FROM_S4_CONFIG_SLOT

// [[Rcpp::export]]
Rcpp::S4 shadow_extract_solver_config(Rcpp::S4 solver_in) {
  Solver& solver = from_S4< Solver >(solver_in);
  return to_S4(solver.config);
}

// [[Rcpp::export]]
Rcpp::S4 shadow_solver_config_new() {
  SolverConfig config;
  return to_S4(config);
}

// [[Rcpp::export(.shadow_get_solver_with_config)]]
Rcpp::S4 shadow_get_solver_with_config(Rcpp::S4 trace_in, Rcpp::S4 config_in) {
  Trace& trace = from_S4<Trace>(trace_in);
  SolverConfig config = shadow_make_solver_config(config_in);
  Solver solver(trace, config);
  return to_S4(solver);
}

// [[Rcpp::export]]
Rcpp::List shadow_extract_solver_states(Rcpp::S4 solver_in) {
  Solver& solver = from_S4< Solver >(solver_in);
  Rcpp::List list;
  for(auto& state : solver.states) {
    Rcpp::S4 state_S4 = to_S4(state);
    list.push_back(state_S4);
  }
  return list;
}

// [[Rcpp::export]]
Rcpp::S4 shadow_solver_maximize(Rcpp::S4 solver_in) {
  Solver& solver = from_S4<Solver>(solver_in).maximize();
  return to_S4(solver);
}































