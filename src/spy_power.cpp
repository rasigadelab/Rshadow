/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "vector_overloads.h"
#include "spy_binary_ops.h"

/*** TENSOR ^ TENSOR ********************************************************/
Tensor pow(const Tensor& a, const Tensor& b) {
	assert(!a.is_null());
	assert(!b.is_null());

	if (a.dim == b.dim) {
		return Tensor(pow(a.val, b.val), a.dim);
	}
	else if (a.is_scalar()) {
		return Tensor(pow(a.val[0], b.val), b.dim);
	}
	else if (b.is_scalar()) {
		return Tensor(pow(a.val, b.val[0]), a.dim);
	}
	else {
		assert(false);
		throw std::logic_error("Uncompatible tensor dimensions");
	}
}

/*** SPY ^ SPY ********************************************************/
Spy pow(const Spy& a, const Spy& b) {
	assert(&a.tape == &b.tape);
	index_t out = -1;

	/* Trivial case a^a */
	if (a.tape_begin() == b.tape_begin()) {
		assert(a.dim == b.dim);
		out = a.is_scalar() ?
			a.tape.rec< SelfPowerScalar >(a) :
			a.tape.rec< SelfPowerVector >(a);
	}else if (a.dim == b.dim) { // Equal dimensions
		out = a.is_scalar() ?
			a.tape.rec< PowerScalarScalar< true, true > >({ a, b }) :
			a.tape.rec< PowerVectorVector< true, true > >({ a, b });
	}
	else { // Unequal dimensions
		if (a.is_scalar()) {
			out = a.tape.rec< PowerScalarVector< true, true > >({ a, b });
		}
		else if (b.is_scalar()) {
			out = a.tape.rec< PowerVectorScalar< true, true > >({ a, b });
		}
		else {
			assert(false);
			throw std::logic_error("Uncompatible tensor dimensions");
		}
	}
	return Spy(pow(Tensor(a), Tensor(b)), a.tape, out);
}

/*** SPY ^ TENSOR ********************************************************/
Spy pow(const Spy& a, const Tensor& b) {
	index_t out = -1;

	if (b.is_scalar() && b.scalar() == -1.) { // Optimize
		out = a.is_scalar() ?
			a.tape.rec< InvertScalar >(a) :
			a.tape.rec< InvertVector >(a);
	}
	else if (b.is_scalar() && b.scalar() == 0.) { // Optimize
		out = a.is_scalar() ?
			a.tape.rec< TrivialScalar< 1 > >(a) :
			a.tape.rec< TrivialVector< 1 > >(a);
	}
	else if (b.is_scalar() && b.scalar() == 1.) { // Optimize
		out = a.is_scalar() ?
			a.tape.rec< IdentityScalar >(a) :
			a.tape.rec< IdentityVector >(a);
	}
	else if (b.is_scalar() && b.scalar() == 2.) { // Optimize
		out = a.is_scalar() ?
			a.tape.rec< SquareScalar >(a) :
			a.tape.rec< SquareVector >(a);
	}
	else if (b.is_scalar() && b.scalar() == 3.) { // Optimize
		out = a.is_scalar() ?
			a.tape.rec< CubeScalar >(a) :
			a.tape.rec< CubeVector >(a);
	}
	else if (a.dim == b.dim) { // Equal dimensions
		out = a.is_scalar() ?
			a.tape.rec< PowerScalarScalar< true, false > >(a, b) :
			a.tape.rec< PowerVectorVector< true, false > >(a, b);
	}
	else { // Unequal dimensions
		if (a.is_scalar()) {
			out = a.tape.rec< PowerScalarVector< true, false > >(a, b);
		}
		else if (b.is_scalar()) {
			out = a.tape.rec< PowerVectorScalar< true, false > >(a, b);
		}
		else {
			assert(false);
			throw std::logic_error("Uncompatible tensor dimensions");
		}
	}
	return Spy(pow(Tensor(a), Tensor(b)), a.tape, out);
}
Spy pow(const Spy& a, const double& b) {
	return pow(a, Tensor(b));
}

/*** TENSOR ^ SPY ********************************************************/
Spy pow(const Tensor& a, const Spy& b) {
	index_t out = -1;
	
	if (a.is_scalar() && a.scalar() == 0.) { // Optimize
		out = b.is_scalar() ?
			b.tape.rec< TrivialScalar< 0 > >(b) :
			b.tape.rec< TrivialVector< 0 > >(b);
	}
	else if (a.is_scalar() && a.scalar() == 1.) { // Optimize
		out = b.is_scalar() ?
			b.tape.rec< TrivialScalar< 1 > >(b) :
			b.tape.rec< TrivialVector< 1 > >(b);
	}
	else if (a.dim == b.dim) { // Equal dimensions
		out = a.is_scalar() ?
			b.tape.rec< PowerScalarScalar< false, true > >(b, a) :
			b.tape.rec< PowerVectorVector< false, true > >(b, a);
	}
	else { // Unequal dimensions
		if (a.is_scalar()) {
			out = b.tape.rec< PowerScalarVector< false, true > >(b, a);
		}
		else if (b.is_scalar()) {
			out = b.tape.rec< PowerVectorScalar< false, true > >(b, a);
		}
		else {
			assert(false);
			throw std::logic_error("Uncompatible tensor dimensions");
		}
	}
	return Spy(pow(Tensor(a), Tensor(b)), b.tape, out);
}
Spy pow(const double& a, const Spy& b) {
	return pow(Tensor(a), b);
}