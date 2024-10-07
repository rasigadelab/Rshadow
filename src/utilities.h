/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once

/* Redefine the assert() macro to break in debugger, include in place of <cassert> */

/* Replace normal assert with breakpoint-triggering assert (NOT PORTABLE!)*/
#include <cassert>

#ifndef NDEBUG
/*
#define assert(expression) (void)(                                                       \
            (!!(expression)) ||                                                              \
            (_wassert(_CRT_WIDE(#expression), _CRT_WIDE(__FILE__), (unsigned)(__LINE__)), 0) \
        )
*/
#include <intrin.h> // __debugbreak() calls, MSVC only

#undef assert
#define assert(EXPRESSION) if(!(EXPRESSION)) __debugbreak(); 

#endif

#include <cmath>
static inline double round(double value, int decimal_places) {
	const double multiplier = std::pow(10.0, decimal_places);
	return std::round(value * multiplier) / multiplier;
}

static inline float round(float value, int decimal_places) {
	const float multiplier = float(std::pow(10.0, decimal_places));
	return std::round(value * multiplier) / multiplier;
}

static inline double log1m(const double& x) {
	return std::log1p(-x);
}

static inline double logit(const double& x) {
	return std::log(x / (1. - x));
}

static inline double logistic(const double& x) {
	return 1. / (1. + std::exp(-x));
}

double inverse_of_normal_cdf(const double p, const double mu, const double sigma);

/* Helper function to generate normal deviates */
#include <vector>
std::vector< double > rnorm(const size_t n, const double mu = 0.0, const double sd = 1.0);


#ifdef DEPREC

/* Index pair of upper triangular matrix, i <= j*/
class unordered_index_pair_t {
public:
	index_t i;
	index_t j;
	unordered_index_pair_t(const index_t i_, const index_t j_) : i{ i_ }, j{ j_ } {
		if (i_ > j_) std::swap(i, j);
	}
	bool operator<(const unordered_index_pair_t& x) const {
		return std::tie(i, j) < std::tie(x.i, x.j);
	}
};

#endif

/*** Explicit index type ***/
using index_t = long long;