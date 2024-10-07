/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "vector_overloads.h"
#include "spy_binary_ops.h"

/*** TENSOR + TENSOR ********************************************************/
Tensor operator+(const Tensor& a, const Tensor& b) {
	assert(!a.is_null());
	assert(!b.is_null());

	if (a.dim == b.dim) {
		return Tensor(a.val + b.val, a.dim);
	}
	else if (a.is_scalar()) {
		return Tensor(a.val + b.val, b.dim);
	}
	else if (b.is_scalar()) {
		return Tensor(a.val + b.val, a.dim);
	}
	else {
		assert(false);
		throw std::logic_error("Uncompatible tensor dimensions");
	}
}

/*** SPY + SPY ********************************************************/
Spy operator+(const Spy& a, const Spy& b) {
	assert(&a.tape == &b.tape);
	index_t out = -1;

	/* Trivial case a/a */
	if (a.tape_begin() == b.tape_begin()) {
		assert(a.dim == b.dim);
		out = a.is_scalar() ?
			a.tape.rec< MultiplyScalarScalar< true, false > >( a, { 2. }) :
			a.tape.rec< MultiplyVectorScalar< true, false > >( a, { 2. });
		return Spy(operator+(Tensor(a), Tensor(b)), a.tape, out);
	} // End trivial case a/a

	/* Non-trivial case */
	if (a.dim == b.dim) { // Equal dimensions
		out = a.is_scalar() ?
			a.tape.rec< PlusScalarScalar< true, true > >({ a, b }) :
			a.tape.rec< PlusVectorVector< true, true > >({ a, b });
	}
	else { // Unequal dimensions
		if (a.is_scalar()) {
			/* Commutative op */
			out = a.tape.rec< PlusVectorScalar< true, true > >({ b, a });
		}
		else if (b.is_scalar()) {
			out = a.tape.rec< PlusVectorScalar< true, true > >({ a, b });
		}
		else {
			assert(false);
			throw std::logic_error("Uncompatible tensor dimensions");
		}
	}
	return Spy(operator+(Tensor(a), Tensor(b)), a.tape, out);
}

/*** SPY + TENSOR ********************************************************/
Spy operator+(const Spy& a, const Tensor& b) {
	index_t out = -1;

	if (b.is_scalar() && b.scalar() == 0.) { // Optimize
		out = a.is_scalar() ?
			a.tape.rec< IdentityScalar >(a) :
			a.tape.rec< IdentityVector >(a);
	}
	else if (a.dim == b.dim) { // Equal dimensions
		out = a.is_scalar() ?
			a.tape.rec< PlusScalarScalar< true, false > >(a, b) :
			a.tape.rec< PlusVectorVector< true, false > >(a, b);
	}
	else { // Unequal dimensions
		if (a.is_scalar()) {
			/* Commutative op */
			out = a.tape.rec< PlusVectorScalar< false, true > >(a, b);
		}
		else if (b.is_scalar()) {
			out = a.tape.rec< PlusVectorScalar< true, false > >(a, b);
		}
		else {
			assert(false);
			throw std::logic_error("Uncompatible tensor dimensions");
		}
	}
	return Spy(operator+(Tensor(a), Tensor(b)), a.tape, out);
}
Spy operator+(const Spy& a, const double& b) {
	return operator+(a, Tensor(b));
}
Spy operator+(const double& a, const Spy& b) {
	return operator+(b, Tensor(a));
}

/*** TENSOR + SPY ********************************************************/
Spy operator+(const Tensor& a, const Spy& b) {
	return operator+(b, a);
}