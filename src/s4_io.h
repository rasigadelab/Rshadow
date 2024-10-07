/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/

#pragma once

/* This file defines common conversion and I/O routines to work with SHADOwC classes
in a Rcpp context. */

#include <Rcpp.h>
#include <shadowc.h>

/* Obtain the const char* name of a S4 class
mapped with the C++ class. The S4 classes are defined in R package source. */
template< typename T > struct S4_name;
template<> struct S4_name<Tensor> { static constexpr char name[] = "shadow_tensor"; };
template<> struct S4_name<Spy>    { static constexpr char name[] = "shadow_spy"; };
template<> struct S4_name<Tape>   { static constexpr char name[] = "shadow_tape"; };
template<> struct S4_name<Trace>  { static constexpr char name[] = "shadow_trace"; };
template<> struct S4_name<Solver> { static constexpr char name[] = "shadow_solver"; };
template<> struct S4_name<SolverConfig> { static constexpr char name[] = "shadow_solver_config"; };
template<> struct S4_name<SolverState>  { static constexpr char name[] = "shadow_solver_state"; };

/****************************************************************************************/
/* EXTERNAL POINTER WRAPPERS */
/****************************************************************************************/

/* The following routines manage C++ pointers, typically SHADOwC objects, in the R context. 
The 'ptr'-containing classes are S4 with a common 'ptr' field. When extracting or creating
C++ objects from S4 objects, the template S4_name< C++_type > provides the S4 class name.

These routines should not be exposed to R (no Rcpp::export)
*/

/* Obtain a C++ reference to the object pointed by slot 'ptr' of a S4 object */
template< typename T >
T& from_S4(Rcpp::S4 x) {
  Rcpp::XPtr<T> ptr = x.slot("ptr");
  if(ptr.get() == nullptr) Rcpp::stop("ptr field was NULL.");
  T& cppx = *(ptr.get());
  return cppx;
}

/* Construct an appropriate S4 object from pre-allocated C++ pointer */
template< typename T >
Rcpp::S4 wrap_ptr_S4(T* ptr, bool allow_nullptr = false) {
  if((!allow_nullptr) && (ptr == nullptr)) {
    Rcpp::stop("wrap_ptr_S4 received a null pointer, possible heap allocation failure.");
  }
  Rcpp::S4 result_S4(S4_name<T>::name);
  Rcpp::XPtr<T> xptr(ptr);
  if(xptr.get() != ptr) Rcpp::stop("Rcpp::XPtr wrapping failed");
  result_S4.slot("ptr") = xptr;
  return result_S4;
}

/* Construct S4 object from C++ object */
template< typename T >
Rcpp::S4 to_S4(T& x);

/* Generic template, assumes that a standard copy constructor is available */
template< typename T >
Rcpp::S4 to_S4(T& x) {
  T* ptr = new T(x);
  if(ptr == nullptr) Rcpp::stop("Heap allocation failed");
  return wrap_ptr_S4(ptr);
}

/* Specializations */
template<> Rcpp::S4 to_S4<Tensor>(Tensor& x);
template<> Rcpp::S4 to_S4<Spy>(Spy& x);
template<> Rcpp::S4 to_S4< SolverConfig >(SolverConfig& config_Cpp);
template<> Rcpp::S4 to_S4<SolverState>(SolverState& state_Cpp);

/* Convert tensor to R numeric format. By default, scalar/vector tensors
  are converted to R vectors without a 'dim' attribute. To always convert
  the dimension vector of the tensor, set force_dim_attr = true. */
Rcpp::NumericVector NumericVector_from_Tensor(const Tensor& tensor, bool force_dim_attr = false);