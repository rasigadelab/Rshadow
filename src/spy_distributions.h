/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/

/**
This file contains probability density functions (PDFs) for common distributions.

The functions can accept arbitrary combinations of scalar constant, vector constants,
or variables (Spy class) using templating idioms. Use the auto keyword to declare
the return type and all intermediate values within functions. The auto keyword in the 
templated context will accomodate either Spy, std::vector< double > or double types
of variables.

The correct return types are propagated through all overloaded functions:
-return type is Spy if at least one argument is a Spy
-return type is std::vector< double > if at least one argument is a vector
-return type is double if all arguments are of type double

**/

#pragma once
#include "spy_binary_ops.h"
#include "spy_unary_ops.h"
#include "spy_aggregator_ops.h"

/*** FUNCTION HEADER MACROS ***/
/* Use these macros to declare spies. The template magic is here to restrict 
* template instantiation to admissible argument types (typically double, Spy, and Tensor).
* Don't forget to use 'auto' for all intermediate return types within function
* body.
*/

#define SHADOW_SPY_1WAY(FUNC_NAME, T1_ARG) \
template< \
	typename T1, \
	typename = std::enable_if_t< Spy::is_spy_arg_v< T1 > > \
> \
static auto FUNC_NAME(const T1& T1_ARG)

#define SHADOW_SPY_2WAY(FUNC_NAME, T1_ARG, T2_ARG) \
template< \
	typename T1, typename T2, \
	typename = std::enable_if_t< Spy::is_spy_arg_v< T1 >&& Spy::is_spy_arg_v< T2 > > \
> \
static auto FUNC_NAME(const T1& T1_ARG, const T2& T2_ARG)

#define SHADOW_SPY_3WAY(FUNC_NAME, T1_ARG, T2_ARG, T3_ARG) \
template< \
	typename T1, typename T2, typename T3, \
	typename = std::enable_if_t< Spy::is_spy_arg_v< T1 >&& Spy::is_spy_arg_v< T2 >&& Spy::is_spy_arg_v< T3 > > \
> \
static auto FUNC_NAME(const T1& T1_ARG, const T2& T2_ARG, const T3& T3_ARG)

#define SHADOW_SPY_4WAY(FUNC_NAME, T1_ARG, T2_ARG, T3_ARG, T4_ARG) \
template< \
	typename T1, typename T2, typename T3, typename T4, \
	typename = std::enable_if_t< Spy::is_spy_arg_v< T1 >&& Spy::is_spy_arg_v< T2 >&& Spy::is_spy_arg_v< T3 >&& Spy::is_spy_arg_v< T4 > > \
> \
static auto FUNC_NAME(const T1& T1_ARG, const T2& T2_ARG, const T3& T3_ARG, const T4& T4_ARG)

#define SHADOW_SPY_5WAY(FUNC_NAME, T1_ARG, T2_ARG, T3_ARG, T4_ARG, T5_ARG) \
template< \
	typename T1, typename T2, typename T3, typename T4, \
	typename = std::enable_if_t< Spy::is_spy_arg_v< T1 >&& Spy::is_spy_arg_v< T2 >&& Spy::is_spy_arg_v< T3 >&& Spy::is_spy_arg_v< T4 >&& Spy::is_spy_arg_v< T5 > > \
> \
static auto FUNC_NAME(const T1& T1_ARG, const T2& T2_ARG, const T3& T3_ARG, const T4& T4_ARG, const T5& T5_ARG)

/****************************************************************************
*** NORMAL DISTRIBUTION
*****************************************************************************/

/* Log-likelihood of the normal distribution with mean mu and standard deviation sd */
SHADOW_SPY_3WAY(logdnorm, x, mu, sd) {
	/* -1/2 * log(2 pi) */
	constexpr double C = -0.918938533204672741780330;
	auto z = (x - mu) / sd;
	return C - 0.5 * pow(z, 2) - log(sd);
};

/****************************************************************************
*** BETA AND DIRICHLET DISTRIBUTIONS
*****************************************************************************/

/* Log-likelihood of the Beta distribution with parameters alpha and beta */
SHADOW_SPY_3WAY(logdbeta, x, alpha, beta) {
	auto normalization_term = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta);
	return (alpha - 1.) * log(x) + (beta - 1.) * log1m(x) + normalization_term;
}

/* Log-likelihood of the Dirichlet distribution with parameter alpha of length >=2.
If x is a vector, it is treated as a single observation. */
SHADOW_SPY_2WAY(logddirichlet, x, alpha) {
	/* Neither x nor alpha can be scalars here */
	static_assert(std::is_same_v< T1, double > == false);
	static_assert(std::is_same_v< T2, double > == false);

	/* For now, both x and alpha must be vectors */
	if constexpr (std::is_same_v< Spy, T1 >) {
		assert(x.dim.size() == 1);
		assert(x.dim[0] > 1);
	}
	else {
		assert(x.size() > 1);
	}
	if constexpr (std::is_same_v< Spy, T2 >) {
		assert(alpha.dim.size() == 1);
	}
	else {
		assert(x.size() > 1);
	}

	auto normalization_term = lgamma(sum(alpha)) - sum(lgamma(alpha));
	return sum((alpha - 1.) * log(x)) + normalization_term;
}

/****************************************************************************
*** UNITARY BETA AND DIRICHLET DISTRIBUTIONS
*****************************************************************************/

/* Log-likelihood of the unitary Beta distribution with parameter mu */
SHADOW_SPY_2WAY(logdunibeta, x, mu) {
	return logdbeta(x, 1. + mu, 2. - mu);
}

/****************************************************************************
*** LOGISTIC DISTRIBUTION
*****************************************************************************/
SHADOW_SPY_2WAY(logdlogis, x, mu) {
	auto mz = mu - x;
	auto expmz = exp(mz);
	return mz - 2. * log1p(expmz);
}

/****************************************************************************
*** TRIGONOMETRIC FUNCTIONS
*****************************************************************************/

SHADOW_SPY_1WAY(tan, x) { return sin(x) / cos(x); }

/****************************************************************************
*** GAMMA DISTRIBUTIONS
*****************************************************************************/

SHADOW_SPY_3WAY(logdgamma, d, alpha, scale) {
	auto bd = d / scale;
	/* Remark the barrier member log(alpha>0) to restrict the domain */
	return log(alpha > 0) + alpha * log(bd) - lgamma(alpha) - log(d) - bd;
}