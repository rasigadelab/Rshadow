/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/

/*
This file contains underlying Rcpp routines to overload (or define)
unary operations on spies and tensors.
*/

#include "s4_io.h"
#include <spy_overloads.h>

template< typename T, typename Tfunc >
Rcpp::S4 shadow_unary_func(Rcpp::S4 in, Tfunc func) {
  T result = func(from_S4<T>(in));
  return to_S4(result);
}

struct functor_negate {
  template< typename T >
  T operator()(const T& in) { return -(in); }
};

#define FUNCTOR_UNARY(FUNC) \
struct functor_##FUNC { \
  template< typename T > \
  T operator()(const T& in) { return FUNC(in); } \
};

// [[Rcpp::export(.shadow_negate_Tensor)]]
Rcpp::S4 shadow_negate_Tensor(Rcpp::S4 x) {
  return shadow_unary_func<Tensor>(x, functor_negate());
}
// [[Rcpp::export(.shadow_negate_Spy)]]
Rcpp::S4 shadow_negate_Spy(Rcpp::S4 x) {
  return shadow_unary_func<Spy>(x, functor_negate());
}

FUNCTOR_UNARY(log);
// [[Rcpp::export(.shadow_log_Tensor)]]
Rcpp::S4 shadow_log_Tensor(Rcpp::S4 x) {
  return shadow_unary_func<Tensor>(x, functor_log());
}
// [[Rcpp::export(.shadow_log_Spy)]]
Rcpp::S4 shadow_log_Spy(Rcpp::S4 x) {
  return shadow_unary_func<Spy>(x, functor_log());
}

FUNCTOR_UNARY(log1p);
// [[Rcpp::export(.shadow_log1p_Tensor)]]
Rcpp::S4 shadow_log1p_Tensor(Rcpp::S4 x) {
  return shadow_unary_func<Tensor>(x, functor_log1p());
}
// [[Rcpp::export(.shadow_log1p_Spy)]]
Rcpp::S4 shadow_log1p_Spy(Rcpp::S4 x) {
  return shadow_unary_func<Spy>(x, functor_log1p());
}

FUNCTOR_UNARY(log1m);
// [[Rcpp::export(.shadow_log1m_Tensor)]]
Rcpp::S4 shadow_log1m_Tensor(Rcpp::S4 x) {
  return shadow_unary_func<Tensor>(x, functor_log1m());
}
// [[Rcpp::export(.shadow_log1m_Spy)]]
Rcpp::S4 shadow_log1m_Spy(Rcpp::S4 x) {
  return shadow_unary_func<Spy>(x, functor_log1m());
}

FUNCTOR_UNARY(exp);
// [[Rcpp::export(.shadow_exp_Tensor)]]
Rcpp::S4 shadow_exp_Tensor(Rcpp::S4 x) {
  return shadow_unary_func<Tensor>(x, functor_exp());
}
// [[Rcpp::export(.shadow_exp_Spy)]]
Rcpp::S4 shadow_exp_Spy(Rcpp::S4 x) {
  return shadow_unary_func<Spy>(x, functor_exp());
}

FUNCTOR_UNARY(lgamma);
// [[Rcpp::export(.shadow_lgamma_Tensor)]]
Rcpp::S4 shadow_lgamma_Tensor(Rcpp::S4 x) {
  return shadow_unary_func<Tensor>(x, functor_lgamma());
}
// [[Rcpp::export(.shadow_lgamma_Spy)]]
Rcpp::S4 shadow_lgamma_Spy(Rcpp::S4 x) {
  return shadow_unary_func<Spy>(x, functor_lgamma());
}

FUNCTOR_UNARY(logit);
// [[Rcpp::export(.shadow_logit_Tensor)]]
Rcpp::S4 shadow_logit_Tensor(Rcpp::S4 x) {
  return shadow_unary_func<Tensor>(x, functor_logit());
}
// [[Rcpp::export(.shadow_logit_Spy)]]
Rcpp::S4 shadow_logit_Spy(Rcpp::S4 x) {
  return shadow_unary_func<Spy>(x, functor_logit());
}

FUNCTOR_UNARY(logistic);
// [[Rcpp::export(.shadow_logistic_Tensor)]]
Rcpp::S4 shadow_logistic_Tensor(Rcpp::S4 x) {
  return shadow_unary_func<Tensor>(x, functor_logistic());
}
// [[Rcpp::export(.shadow_logistic_Spy)]]
Rcpp::S4 shadow_logistic_Spy(Rcpp::S4 x) {
  return shadow_unary_func<Spy>(x, functor_logistic());
}

FUNCTOR_UNARY(sum);
// [[Rcpp::export(.shadow_sum_Tensor)]]
Rcpp::S4 shadow_sum_Tensor(Rcpp::S4 x) {
  return shadow_unary_func<Tensor>(x, functor_sum());
}
// [[Rcpp::export(.shadow_sum_Spy)]]
Rcpp::S4 shadow_sum_Spy(Rcpp::S4 x) {
  return shadow_unary_func<Spy>(x, functor_sum());
}

FUNCTOR_UNARY(sumsq);
// [[Rcpp::export(.shadow_sumsq_Tensor)]]
Rcpp::S4 shadow_sumsq_Tensor(Rcpp::S4 x) {
  return shadow_unary_func<Tensor>(x, functor_sumsq());
}
// [[Rcpp::export(.shadow_sumsq_Spy)]]
Rcpp::S4 shadow_sumsq_Spy(Rcpp::S4 x) {
  return shadow_unary_func<Spy>(x, functor_sumsq());
}

#undef FUNCTOR_UNARY