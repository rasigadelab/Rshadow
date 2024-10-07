/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include <vector>
#include <cmath>
#include "utilities.h"

/* This file contains overloads of standard functions and operators for std::vector< double >.
Be certain to encapsulate in namespace later on or use a specific vector class.
Currently the header is only called from .cpp compilation units and not exposed. */

/******************************************************************
*** EQUALITY OPERATOR OVERLOADS *************************************
*******************************************************************/

template< typename T >
bool operator==(const std::vector< T >& a, const std::vector< T >& b) {
	if (a.size() != b.size()) return false;
	for (size_t i = 0; i < a.size(); i++) {
		if (a[i] != b[i]) return false;
	}
	return true;
}

/******************************************************************
*** BINARY OPERATOR OVERLOADS *************************************
*******************************************************************/

#define SHADOW_VECTOR_BINARY_OPERATOR_OVERLOAD(OP) \
static inline std::vector< double > operator OP(const double& a, const std::vector< double >& b) { \
std::vector< double > y(b.size()); \
for (size_t i = 0; i < b.size(); i++) { \
	y[i] = a OP b[i]; \
} \
return y; \
} \
static inline std::vector< double > operator OP(const std::vector< double >& a, const double& b) { \
	std::vector< double > y(a.size()); \
	for (size_t i = 0; i < a.size(); i++) { \
		y[i] = a[i] OP b; \
	} \
	return y; \
} \
static inline std::vector< double > operator OP(const std::vector< double >& a, const std::vector< double >& b) { \
	if (a.size() == 1) return a[0] OP b; \
	if (b.size() == 1) return a OP b[0]; \
	std::vector< double > y(a.size()); \
	assert(a.size() == b.size()); \
	for (size_t i = 0; i < y.size(); i++) { \
		y[i] = a[i] OP b[i]; \
	} \
	return y; \
}

SHADOW_VECTOR_BINARY_OPERATOR_OVERLOAD(+);
SHADOW_VECTOR_BINARY_OPERATOR_OVERLOAD(-);
SHADOW_VECTOR_BINARY_OPERATOR_OVERLOAD(*);
SHADOW_VECTOR_BINARY_OPERATOR_OVERLOAD(/);
SHADOW_VECTOR_BINARY_OPERATOR_OVERLOAD(>);
SHADOW_VECTOR_BINARY_OPERATOR_OVERLOAD(>=);
SHADOW_VECTOR_BINARY_OPERATOR_OVERLOAD(<);
SHADOW_VECTOR_BINARY_OPERATOR_OVERLOAD(<=);

#undef SHADOW_VECTOR_BINARY_OPERATOR_OVERLOAD

/******************************************************************
*** UNARY OPERATOR OVERLOADS **************************************
*******************************************************************/

/* No macro here as long as only operator- is overloaded */

static inline std::vector< double > operator-(const std::vector< double >& a) {
	std::vector< double > y(a.size());
	for (size_t i = 0; i < a.size(); i++) {
		y[i] = -a[i];
	}
	return y;
}

/******************************************************************
*** BINARY FUNCTION OVERLOADS *************************************
*******************************************************************/

#define SHADOW_VECTOR_BINARY_FUNCTION_OVERLOAD(FUNC) \
static inline std::vector< double > FUNC(const double& a, const std::vector< double >& b) { \
std::vector< double > y(b.size()); \
for (size_t i = 0; i < b.size(); i++) { \
	y[i] = FUNC(a, b[i]); \
} \
return y; \
} \
static inline std::vector< double > FUNC(const std::vector< double >& a, const double& b) { \
	std::vector< double > y(a.size()); \
	for (size_t i = 0; i < a.size(); i++) { \
		y[i] = FUNC(a[i], b); \
	} \
	return y; \
} \
static inline std::vector< double > FUNC(const std::vector< double >& a, const std::vector< double >& b) { \
	if (a.size() == 1) return FUNC(a[0], b); \
	if (b.size() == 1) return FUNC(a, b[0]); \
	std::vector< double > y(a.size()); \
	assert(a.size() == b.size()); \
	for (size_t i = 0; i < y.size(); i++) { \
		y[i] = FUNC(a[i], b[i]); \
	} \
	return y; \
}

SHADOW_VECTOR_BINARY_FUNCTION_OVERLOAD(pow);

#undef SHADOW_VECTOR_BINARY_FUNCTION_OVERLOAD

/******************************************************************
*** UNARY FUNCTION OVERLOADS **************************************
*******************************************************************/

#define SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(FUNC) \
static inline std::vector< double > FUNC(const std::vector< double >& a) { \
std::vector< double > y(a.size()); \
for (size_t i = 0; i < a.size(); i++) { \
	y[i] = FUNC(a[i]); \
	} \
	return y; \
};

SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(log);
SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(log1p);
SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(log1m);
SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(exp);
SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(lgamma);
SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(logit);
SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(logistic);
SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(sin);
SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(cos);
SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD(tan);

#undef SHADOW_VECTOR_UNARY_FUNCTION_OVERLOAD