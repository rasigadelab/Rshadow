/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/

/*
This file contains underlying Rcpp routines to overload (or define)
binary operations on spies and tensors.
*/

#include "s4_io.h"
#include <spy_overloads.h>
#include <functional>

/* Obtain the output type of a binary operator */
template< typename Tleft, typename Tright> struct ResultType;
template<> struct ResultType< Tensor, Tensor > { using type = Tensor; };
template<> struct ResultType< Tensor, Spy >    { using type = Spy; };
template<> struct ResultType< Spy,    Tensor > { using type = Spy; };
template<> struct ResultType< Spy,    Spy >    { using type = Spy; };

/*** OPERATOR TEMPLATE ******************************************/

/* Provide a binary functor such as std::plus() to apply the 
operation to a pair of spies or tensors. See examples below.
*/

template< typename Tleft, typename Tright, typename Op>
Rcpp::S4 shadow_operator(Rcpp::S4 left, Rcpp::S4 right, Op op) {
  using Tresult = typename ResultType< Tleft, Tright >::type;
  Tresult result = op(from_S4<Tleft>(left), from_S4<Tright>(right));
  return to_S4(result);
}

/*** PLUS ******************************************/

// [[Rcpp::export(.shadow_plus_Tensor_Tensor)]]
Rcpp::S4 shadow_plus_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Tensor, Tensor>(left, right, std::plus());
}
// [[Rcpp::export(.shadow_plus_Tensor_Spy)]]
Rcpp::S4 shadow_plus_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Tensor, Spy>(left, right, std::plus());
}
// [[Rcpp::export(.shadow_plus_Spy_Tensor)]]
Rcpp::S4 shadow_plus_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Spy, Tensor>(left, right, std::plus());
}
// [[Rcpp::export(.shadow_plus_Spy_Spy)]]
Rcpp::S4 shadow_plus_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Spy, Spy>(left, right, std::plus());
}

/*** MINUS ******************************************/

// [[Rcpp::export(.shadow_minus_Tensor_Tensor)]]
Rcpp::S4 shadow_minus_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Tensor, Tensor>(left, right, std::minus());
}
// [[Rcpp::export(.shadow_minus_Tensor_Spy)]]
Rcpp::S4 shadow_minus_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Tensor, Spy>(left, right, std::minus());
}
// [[Rcpp::export(.shadow_minus_Spy_Tensor)]]
Rcpp::S4 shadow_minus_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Spy, Tensor>(left, right, std::minus());
}
// [[Rcpp::export(.shadow_minus_Spy_Spy)]]
Rcpp::S4 shadow_minus_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Spy, Spy>(left, right, std::minus());
}

/*** MULTIPLIES ******************************************/

// [[Rcpp::export(.shadow_multiplies_Tensor_Tensor)]]
Rcpp::S4 shadow_multiplies_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Tensor, Tensor>(left, right, std::multiplies());
}
// [[Rcpp::export(.shadow_multiplies_Tensor_Spy)]]
Rcpp::S4 shadow_multiplies_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Tensor, Spy>(left, right, std::multiplies());
}
// [[Rcpp::export(.shadow_multiplies_Spy_Tensor)]]
Rcpp::S4 shadow_multiplies_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Spy, Tensor>(left, right, std::multiplies());
}
// [[Rcpp::export(.shadow_multiplies_Spy_Spy)]]
Rcpp::S4 shadow_multiplies_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Spy, Spy>(left, right, std::multiplies());
}

/*** DIVIDES ******************************************/

// [[Rcpp::export(.shadow_divides_Tensor_Tensor)]]
Rcpp::S4 shadow_divides_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Tensor, Tensor>(left, right, std::divides());
}
// [[Rcpp::export(.shadow_divides_Tensor_Spy)]]
Rcpp::S4 shadow_divides_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Tensor, Spy>(left, right, std::divides());
}
// [[Rcpp::export(.shadow_divides_Spy_Tensor)]]
Rcpp::S4 shadow_divides_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Spy, Tensor>(left, right, std::divides());
}
// [[Rcpp::export(.shadow_divides_Spy_Spy)]]
Rcpp::S4 shadow_divides_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_operator<Spy, Spy>(left, right, std::divides());
}

/*** LESS THAN ******************************************/

// [[Rcpp::export(.shadow_less_Tensor_Tensor)]]
Rcpp::S4 shadow_less_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Tensor, Tensor>(left, right, std::less());
}
// [[Rcpp::export(.shadow_less_Tensor_Spy)]]
Rcpp::S4 shadow_less_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Tensor, Spy>(left, right, std::less());
}
// [[Rcpp::export(.shadow_less_Spy_Tensor)]]
Rcpp::S4 shadow_less_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Spy, Tensor>(left, right, std::less());
}
// [[Rcpp::export(.shadow_less_Spy_Spy)]]
Rcpp::S4 shadow_less_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Spy, Spy>(left, right, std::less());
}

/*** LESS THAN OR EQUAL **************************************/

// [[Rcpp::export(.shadow_less_equal_Tensor_Tensor)]]
Rcpp::S4 shadow_less_equal_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Tensor, Tensor>(left, right, std::less_equal());
}
// [[Rcpp::export(.shadow_less_equal_Tensor_Spy)]]
Rcpp::S4 shadow_less_equal_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Tensor, Spy>(left, right, std::less_equal());
}
// [[Rcpp::export(.shadow_less_equal_Spy_Tensor)]]
Rcpp::S4 shadow_less_equal_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Spy, Tensor>(left, right, std::less_equal());
}
// [[Rcpp::export(.shadow_less_equal_Spy_Spy)]]
Rcpp::S4 shadow_less_equal_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Spy, Spy>(left, right, std::less_equal());
}

/*** GREATER THAN ******************************************/

// [[Rcpp::export(.shadow_greater_Tensor_Tensor)]]
Rcpp::S4 shadow_greater_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Tensor, Tensor>(left, right, std::greater());
}
// [[Rcpp::export(.shadow_greater_Tensor_Spy)]]
Rcpp::S4 shadow_greater_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Tensor, Spy>(left, right, std::greater());
}
// [[Rcpp::export(.shadow_greater_Spy_Tensor)]]
Rcpp::S4 shadow_greater_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Spy, Tensor>(left, right, std::greater());
}
// [[Rcpp::export(.shadow_greater_Spy_Spy)]]
Rcpp::S4 shadow_greater_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Spy, Spy>(left, right, std::greater());
}

/*** GREATER THAN OR EQUAL **************************************/

// [[Rcpp::export(.shadow_greater_equal_Tensor_Tensor)]]
Rcpp::S4 shadow_greater_equal_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Tensor, Tensor>(left, right, std::greater_equal());
}
// [[Rcpp::export(.shadow_greater_equal_Tensor_Spy)]]
Rcpp::S4 shadow_greater_equal_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Tensor, Spy>(left, right, std::greater_equal());
}
// [[Rcpp::export(.shadow_greater_equal_Spy_Tensor)]]
Rcpp::S4 shadow_greater_equal_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Spy, Tensor>(left, right, std::greater_equal());
}
// [[Rcpp::export(.shadow_greater_equal_Spy_Spy)]]
Rcpp::S4 shadow_greater_equal_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
    return shadow_operator<Spy, Spy>(left, right, std::greater_equal());
}

/*** BINARY FUNCTION TEMPLATE ******************************************/

template< typename Tleft, typename Tright, typename Tfunc, typename Tresult = typename ResultType< Tleft, Tright >::type>
Rcpp::S4 shadow_binary_func(Rcpp::S4 left, Rcpp::S4 right, Tfunc func) {
  Tresult result = func(from_S4<Tleft>(left), from_S4<Tright>(right));
  return to_S4(result);
}

/* This macro defines a functor with the name of the provided binary function,
such as pow(x, y) */
#define FUNCTOR_BINARY(FUNC) \
struct functor_##FUNC { \
  template< typename Tleft, typename Tright, \
    typename Tresult = typename ResultType< Tleft, Tright >::type> \
  Tresult operator()(const Tleft& left, const Tright& right) { \
    return FUNC(left, right); \
  } \
};

/*** POWER ******************************************/

FUNCTOR_BINARY(pow)
// [[Rcpp::export(.shadow_pow_Tensor_Tensor)]]
Rcpp::S4 shadow_pow_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Tensor, Tensor>(left, right, functor_pow());
}
// [[Rcpp::export(.shadow_pow_Tensor_Spy)]]
Rcpp::S4 shadow_pow_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Tensor, Spy>(left, right, functor_pow());
}
// [[Rcpp::export(.shadow_pow_Spy_Tensor)]]
Rcpp::S4 shadow_pow_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Spy, Tensor>(left, right, functor_pow());
}
// [[Rcpp::export(.shadow_pow_Spy_Spy)]]
Rcpp::S4 shadow_pow_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Spy, Spy>(left, right, functor_pow());
}

/*** DOT PRODUCT ******************************************/

// Tensor/Spoy and Spy/Tensor still missing in SHADOwC

FUNCTOR_BINARY(dot)
// [[Rcpp::export(.shadow_dot_Tensor_Tensor)]]
Rcpp::S4 shadow_dot_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Tensor, Tensor>(left, right, functor_dot());
}
// [[Rcpp::export(.shadow_dot_Tensor_Spy)]]
Rcpp::S4 shadow_dot_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Tensor, Spy>(left, right, functor_dot());
}
// [[Rcpp::export(.shadow_dot_Spy_Tensor)]]
Rcpp::S4 shadow_dot_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Spy, Tensor>(left, right, functor_dot());
}
// [[Rcpp::export(.shadow_dot_Spy_Spy)]]
Rcpp::S4 shadow_dot_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Spy, Spy>(left, right, functor_dot());
}

/*** SUM LOG BERN ******************************************/

FUNCTOR_BINARY(sum_log_dbern)
// [[Rcpp::export(.shadow_sumlogbern_Tensor_Tensor)]]
Rcpp::S4 shadow_sumlogbern_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Tensor, Tensor>(left, right, functor_sum_log_dbern());
}
// [[Rcpp::export(.shadow_sumlogbern_Spy_Tensor)]]
Rcpp::S4 shadow_sumlogbern_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Spy, Tensor>(left, right, functor_sum_log_dbern());
}

/*** MATRIX PRODUCT ******************************************/

FUNCTOR_BINARY(matmult)
// [[Rcpp::export(.shadow_matmult_Tensor_Tensor)]]
Rcpp::S4 shadow_matmult_Tensor_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Tensor, Tensor>(left, right, functor_matmult());
}
// [[Rcpp::export(.shadow_matmult_Tensor_Spy)]]
Rcpp::S4 shadow_matmult_Tensor_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Tensor, Spy>(left, right, functor_matmult());
}
// [[Rcpp::export(.shadow_matmult_Spy_Tensor)]]
Rcpp::S4 shadow_matmult_Spy_Tensor(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Spy, Tensor>(left, right, functor_matmult());
}
// [[Rcpp::export(.shadow_matmult_Spy_Spy)]]
Rcpp::S4 shadow_matmult_Spy_Spy(Rcpp::S4 left, Rcpp::S4 right) {
  return shadow_binary_func<Spy, Spy>(left, right, functor_matmult());
}

#undef FUNCTOR_BINARY
