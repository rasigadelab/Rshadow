/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "vector_overloads.h"
#include "spy_binary_ops.h"

/* STATIC ROUTINES */
static Spy greater_than_zero(const Spy& x) {
	index_t out = x.is_scalar() ?
		x.tape.rec< GreaterThanZeroScalar >(x) :
		x.tape.rec< GreaterThanZeroVector >(x);
	return Spy(operator>(Tensor(x), Tensor(0.0)), x.tape, out);
}
static Spy greater_than_or_equal_zero(const Spy& x) {
	index_t out = x.is_scalar() ?
		x.tape.rec< GreaterThanOrEqualZeroScalar >(x) :
		x.tape.rec< GreaterThanOrEqualZeroVector >(x);
	return Spy(operator>=(Tensor(x), Tensor(0.0)), x.tape, out);
}

/****************************************************************************
*** GREATER THAN
*****************************************************************************/

/*** TENSOR / TENSOR ********************************************************/
Tensor operator>(const Tensor& a, const Tensor& b) {
	assert(!a.is_null());
	assert(!b.is_null());

	if ((a.dim == b.dim) || a.is_scalar() || b.is_scalar()) {
		return Tensor(a.val > b.val, a.is_scalar() ? b.dim : a.dim);
	}
	else {	
		throw std::logic_error("Uncompatible tensor dimensions");
	}
}

/*** SPY / SPY ********************************************************/
Spy operator>(const Spy& a, const Spy& b) {
	assert(&a.tape == &b.tape);
	
	/* Trivial case [a > a] = 0 */
	if (a.tape_begin() == b.tape_begin()) {
		assert(a.dim == b.dim);
		index_t out = a.is_scalar() ?
			a.tape.rec< TrivialScalar<0> >(a) :
			a.tape.rec< TrivialVector<0> >(a);
		return Spy(Tensor(a.dim).fill(0.), a.tape, out);
	} // End trivial case [a > a]

	return greater_than_zero(a - b);
}

/*** SPY / TENSOR ********************************************************/
Spy operator>(const Spy& a, const Tensor& b) {
	return greater_than_zero(a - b);
}
Spy operator>(const Spy& a, const double& b) {
	return operator>(a, Tensor(b));
}

/*** TENSOR / SPY ********************************************************/
Spy operator>(const Tensor& a, const Spy& b) {
	return greater_than_zero(a - b);
}
Spy operator>(const double& a, const Spy& b) {
	return operator>(Tensor(a), b);
}

/****************************************************************************
*** GREATER THAN OR EQUAL
*****************************************************************************/

/*** TENSOR / TENSOR ********************************************************/
Tensor operator>=(const Tensor& a, const Tensor& b) {
	assert(!a.is_null());
	assert(!b.is_null());

	if ((a.dim == b.dim) || a.is_scalar() || b.is_scalar()) {
		return Tensor(a.val >= b.val, a.is_scalar() ? b.dim : a.dim);
	}
	else {
		throw std::logic_error("Uncompatible tensor dimensions");
	}
}

/*** SPY / SPY ********************************************************/
Spy operator>=(const Spy& a, const Spy& b) {
	assert(&a.tape == &b.tape);

	/* Trivial case [a >= a] = 1 */
	if (a.tape_begin() == b.tape_begin()) {
		assert(a.dim == b.dim);
		index_t out = a.is_scalar() ?
			a.tape.rec< TrivialScalar<1> >(a) :
			a.tape.rec< TrivialVector<1> >(a);
		return Spy(Tensor(a.dim).fill(1.), a.tape, out);
	} // End trivial case [a >= a]

	return greater_than_or_equal_zero(a - b);
}

/*** SPY / TENSOR ********************************************************/
Spy operator>=(const Spy& a, const Tensor& b) {
	return greater_than_or_equal_zero(a - b);
}
Spy operator>=(const Spy& a, const double& b) {
	return operator>=(a, Tensor(b));
}

/*** TENSOR / SPY ********************************************************/
Spy operator>=(const Tensor& a, const Spy& b) {
	return greater_than_or_equal_zero(a - b);
}
Spy operator>=(const double& a, const Spy& b) {
	return operator>=(Tensor(a), b);
}

/****************************************************************************
*** LESS THAN
*****************************************************************************/

/*** TENSOR / TENSOR ********************************************************/
Tensor operator<(const Tensor& a, const Tensor& b) {
	assert(!a.is_null());
	assert(!b.is_null());

	if ((a.dim == b.dim) || a.is_scalar() || b.is_scalar()) {
		return Tensor(a.val < b.val, a.is_scalar() ? b.dim : a.dim);
	}
	else {
		throw std::logic_error("Uncompatible tensor dimensions");
	}
}

/*** SPY / SPY ********************************************************/
Spy operator<(const Spy& a, const Spy& b) {
	assert(&a.tape == &b.tape);

	/* Trivial case [a < a] = 0 */
	if (a.tape_begin() == b.tape_begin()) {
		assert(a.dim == b.dim);
		index_t out = a.is_scalar() ?
			a.tape.rec< TrivialScalar<0> >(a) :
			a.tape.rec< TrivialVector<0> >(a);
		return Spy(Tensor(a.dim).fill(0.), a.tape, out);
	} // End trivial case [a < a]

	return greater_than_zero(b - a);
}

/*** SPY / TENSOR ********************************************************/
Spy operator<(const Spy& a, const Tensor& b) {
	return greater_than_zero(b - a);
}
Spy operator<(const Spy& a, const double& b) {
	return operator<(a, Tensor(b));
}

/*** TENSOR / SPY ********************************************************/
Spy operator<(const Tensor& a, const Spy& b) {
	return greater_than_zero(b - a);
}
Spy operator<(const double& a, const Spy& b) {
	return operator<(Tensor(a), b);
}

/****************************************************************************
*** LESS THAN OR EQUAL
*****************************************************************************/

/*** TENSOR / TENSOR ********************************************************/
Tensor operator<=(const Tensor& a, const Tensor& b) {
	assert(!a.is_null());
	assert(!b.is_null());

	if ((a.dim == b.dim) || a.is_scalar() || b.is_scalar()) {
		return Tensor(a.val <= b.val, a.is_scalar() ? b.dim : a.dim);
	}
	else {
		throw std::logic_error("Uncompatible tensor dimensions");
	}
}

/*** SPY / SPY ********************************************************/
Spy operator<=(const Spy& a, const Spy& b) {
	assert(&a.tape == &b.tape);

	/* Trivial case [a <= a] = 1 */
	if (a.tape_begin() == b.tape_begin()) {
		assert(a.dim == b.dim);
		index_t out = a.is_scalar() ?
			a.tape.rec< TrivialScalar<1> >(a) :
			a.tape.rec< TrivialVector<1> >(a);
		return Spy(Tensor(a.dim).fill(1.), a.tape, out);
	} // End trivial case [a <= a]

	return greater_than_or_equal_zero(b - a);
}

/*** SPY / TENSOR ********************************************************/
Spy operator<=(const Spy& a, const Tensor& b) {
	return greater_than_or_equal_zero(b - a);
}
Spy operator<=(const Spy& a, const double& b) {
	return operator<=(a, Tensor(b));
}

/*** TENSOR / SPY ********************************************************/
Spy operator<=(const Tensor& a, const Spy& b) {
	return greater_than_or_equal_zero(b - a);
}
Spy operator<=(const double& a, const Spy& b) {
	return operator<=(Tensor(a), b);
}