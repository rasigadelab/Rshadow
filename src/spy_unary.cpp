/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "vector_overloads.h"
#include "spy_unary_ops.h"


#define SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(FUNC) \
Tensor FUNC(const Tensor& a) { \
	Tensor y(a.dim); \
	y.val = FUNC(a.val); \
	return y; \
}

/*** UNARY NEGATION **********************************************/
SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(operator-);

Spy operator-(const Spy& a) {
	assert(a.is_null() == false);
	index_t out = a.is_scalar() ?
		a.tape.rec< NegateScalar >(a) :
		a.tape.rec< NegateVector >(a);
	return Spy{ operator-(Tensor(a)), a.tape, out };
}

/*** NATURAL LOG *************************************************/

SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(log);

Spy log(const Spy& a) {
	assert(a.is_null() == false);
	index_t out = a.is_scalar() ?
		a.tape.rec< LogScalar >(a) :
		a.tape.rec< LogVector >(a);
	return Spy{ log(Tensor(a)), a.tape, out };
}
/*** LOG(1 + X) **************************************************/
SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(log1p);

Spy log1p(const Spy& a) {
	assert(a.is_null() == false);
	index_t out = a.is_scalar() ?
		a.tape.rec< Log1pScalar >(a) :
		a.tape.rec< Log1pVector >(a);
	return Spy{ log1p(Tensor(a)), a.tape, out};
}

/*** LOG(1 - X) **************************************************/
SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(log1m);

Spy log1m(const Spy& a) {
	assert(a.is_null() == false);
	index_t out = a.is_scalar() ?
		a.tape.rec< Log1mScalar >(a) :
		a.tape.rec< Log1mVector >(a);
	return Spy{ log1m(Tensor(a)), a.tape, out };
}
/*** EXP *********************************************************/
SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(exp);

Spy exp(const Spy& a) {
	assert(a.is_null() == false);
	index_t out = a.is_scalar() ?
		a.tape.rec< ExpScalar >(a) :
		a.tape.rec< ExpVector >(a);
	return Spy{ exp(Tensor(a)), a.tape, out };
}
/*** LGAMMA *********************************************************/
SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(lgamma);

Spy lgamma(const Spy& a) {
	assert(a.is_null() == false);
	index_t out = a.is_scalar() ?
		a.tape.rec< LogGammaScalar >(a) :
		a.tape.rec< LogGammaVector >(a);
	return Spy{ lgamma(Tensor(a)), a.tape, out };
}
/*** LOGIT *********************************************************/
SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(logit);

Spy logit(const Spy& a) {
	assert(a.is_null() == false);
	index_t out = a.is_scalar() ?
		a.tape.rec< LogitScalar >(a) :
		a.tape.rec< LogitVector >(a);
	return Spy{ logit(Tensor(a)), a.tape, out };
}
/*** LOGISTIC *****************************************************/
SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(logistic);

Spy logistic(const Spy& a) {
	assert(a.is_null() == false);
	index_t out = a.is_scalar() ?
		a.tape.rec< LogisticScalar >(a) :
		a.tape.rec< LogisticVector >(a);
	return Spy{ logistic(Tensor(a)), a.tape, out };
}
/*** SIN *****************************************************/
SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(sin);

Spy sin(const Spy& a) {
	assert(a.is_null() == false);
	index_t out = a.is_scalar() ?
		a.tape.rec< SinScalar >(a) :
		a.tape.rec< SinVector >(a);
	return Spy{ sin(Tensor(a)), a.tape, out };
}
/*** COS *****************************************************/
SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD(cos);

Spy cos(const Spy& a) {
	assert(a.is_null() == false);
	index_t out = a.is_scalar() ?
		a.tape.rec< CosScalar >(a) :
		a.tape.rec< CosVector >(a);
	return Spy{ cos(Tensor(a)), a.tape, out };
}


#undef SHADOW_TENSOR_UNARY_FUNCTION_OVERLOAD