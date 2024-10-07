/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "vector_overloads.h"
#include "spy_binary_ops.h"

/*** TENSOR / TENSOR ********************************************************/
Tensor operator/(const Tensor& a, const Tensor& b) {
	assert(!a.is_null());
	assert(!b.is_null());
	
	if (a.dim == b.dim) {
		return Tensor(a.val / b.val, a.dim);
	}
	else if (a.is_scalar()) {
		return Tensor(a.val / b.val, b.dim);
	}
	else if (b.is_scalar()) {
		return Tensor(a.val / b.val, a.dim);
	}
	else {
		assert(false);
		throw std::logic_error("Uncompatible tensor dimensions");	
	}
}

/*** SPY / SPY ********************************************************/
Spy operator/(const Spy& a, const Spy& b) {
	assert(&a.tape == &b.tape);
	index_t out = -1;

	/* Trivial case a/a */
	if (a.tape_begin() == b.tape_begin()) {
		assert(a.dim == b.dim);
		out = a.is_scalar() ?
			a.tape.rec< TrivialScalar<1> >(a) :
			a.tape.rec< TrivialVector<1> >(a);
		return Spy( Tensor(a.dim).fill(1.), a.tape, out);
	} // End trivial case a/a

	/* Non-trivial case */
	if (a.dim == b.dim) { // Equal dimensions
		out = a.is_scalar() ?
			a.tape.rec< DivideScalarScalar< true, true > >({ a, b }) :
			a.tape.rec< DivideVectorVector< true, true > >({ a, b });
	}
	else { // Unequal dimensions
		if (a.is_scalar()) {
			out = a.tape.rec< DivideScalarVector< true, true > >({ a, b });
		}
		else if (b.is_scalar()) {
			out = a.tape.rec< DivideVectorScalar< true, true > >({ a, b });
		}
		else {
			assert(false);
			throw std::logic_error("Uncompatible tensor dimensions");
		}
	}
	return Spy(operator/(Tensor(a), Tensor(b)), a.tape, out);
}
 
/*** SPY / TENSOR ********************************************************/
Spy operator/(const Spy& a, const Tensor& b) {
	index_t out = -1;

	if (b.is_scalar() && b.scalar() == 1.) { // Optimize
		out = a.is_scalar() ?
			a.tape.rec< IdentityScalar >(a) :
			a.tape.rec< IdentityVector >(a);
	}
	else if (a.dim == b.dim) { // Equal dimensions
		out = a.is_scalar() ?
			a.tape.rec< DivideScalarScalar< true, false > >(a, b) :
			a.tape.rec< DivideVectorVector< true, false > >(a, b);
	}
	else { // Unequal dimensions
		if (a.is_scalar()) {
			out = a.tape.rec< DivideScalarVector< true, false > >(a, b);
		}
		else if (b.is_scalar()) {
			out = a.tape.rec< DivideVectorScalar< true, false > >(a, b);
		}
		else {
			assert(false);
			throw std::logic_error("Uncompatible tensor dimensions");
		}
	}
	return Spy(operator/(Tensor(a), Tensor(b)), a.tape, out);
}
Spy operator/(const Spy& a, const double& b) {
	return operator/(a, Tensor(b));
}

/*** TENSOR / SPY ********************************************************/
Spy operator/(const Tensor& a, const Spy& b) {
	index_t out = -1;

	if (a.is_scalar() && a.scalar() == 0.) { // Optimize
		out = b.is_scalar() ?
			b.tape.rec< TrivialScalar<0> >(b) :
			b.tape.rec< TrivialVector<0> >(b);
	}
	else if (a.is_scalar() && a.scalar() == 1.) { // Optimize
		out = b.is_scalar() ?
			b.tape.rec< InvertScalar >(b) :
			b.tape.rec< InvertVector >(b);
	}
	else if (a.dim == b.dim) { // Equal dimensions
		out = a.is_scalar() ?
			b.tape.rec< DivideScalarScalar< false, true > >(b, a) :
			b.tape.rec< DivideVectorVector< false, true > >(b, a);
	}
	else { // Unequal dimensions
		if (a.is_scalar()) {
			out = b.tape.rec< DivideScalarVector< false, true > >(b, a);
		}
		else if (b.is_scalar()) {
			out = b.tape.rec< DivideVectorScalar< false, true > >(b, a);
		}
		else {
			assert(false);
			throw std::logic_error("Uncompatible tensor dimensions");
		}
	}
	return Spy(operator/(Tensor(a), Tensor(b)), b.tape, out);
}
Spy operator/(const double& a, const Spy& b) {
	return operator/(Tensor(a), b);
}

