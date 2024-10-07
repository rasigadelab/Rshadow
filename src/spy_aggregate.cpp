/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/

#include "spy_aggregator_ops.h"

Tensor sum(const Tensor& a) {
	double y = 0.0;
	for (const auto& x : a.val) y += x;
	return y;
}
Spy sum(const Spy& a) {
	index_t out = a.tape.rec< AggregSum >(a);
	return Spy{ sum(Tensor(a)), a.tape, out };
}
Tensor sumsq(const Tensor& a) {
	double y = 0.0;
	for (const auto& x : a.val) y += x * x;
	return y;
}
Spy sumsq(const Spy& a) {
	index_t out = a.tape.rec< AggregSumOfSquares >(a);
	return Spy{ sumsq(Tensor(a)), a.tape, out };
}
Tensor dot(const Tensor& a, const Tensor& b) {
	assert(a.val.size() == b.val.size());
	double y = 0.0;
	for (size_t i = 0; i < a.val.size(); i++) y += a.val[i] * b.val[i];
	return y;
}
Spy dot(const Spy& a, const Spy& b) {
	assert(&a.tape == &b.tape);
	assert(a.val.size() == b.val.size());

	if (a.tape_begin() == b.tape_begin()) {
		return sumsq(a);
	}	
	index_t out = a.is_scalar() ?
		a.tape.rec< MultiplyScalarScalar< true, true > >({ a, b }) :
		a.tape.rec< AggregDotProd<> >({ a, b });
	return Spy{ dot(Tensor(a), Tensor(b)) , a.tape, out};
}
Spy dot(const Spy& a, const Tensor& b) {
	assert(a.val.size() == b.val.size());

	index_t out = a.is_scalar() ?
		a.tape.rec< MultiplyScalarScalar< true, false > >( a, b) :
		a.tape.rec< AggregDotProd<true, false> >( a, b);
	return Spy{ dot(Tensor(a), Tensor(b)) , a.tape, out };
}
Spy dot(const Tensor& a, const Spy& b) {
	assert(a.val.size() == b.val.size());

	index_t out = a.is_scalar() ?
		b.tape.rec< MultiplyScalarScalar< false, true > >(b, a) :
		b.tape.rec< AggregDotProd<false, true> >(b, a);
	return Spy{ dot(Tensor(a), Tensor(b)) , b.tape, out };
}

Tensor sum_log_dbern(const Tensor& a, const Tensor& b) {
	assert(a.val.size() == b.size());
	assert(a.is_vector() && b.is_vector());
	double y = 0.0;
	for (size_t i = 0; i < a.val.size(); i++) {
		y += b.val[i] ? std::log(a.val[i]) : std::log1p(-a.val[i]);
	}
	return y;
}
Spy sum_log_dbern(const Spy& a, const Tensor& b) {	
	assert(a.val.size() == b.size());

#ifdef TOO_SLOW
#ifndef NDEBUG
	/* Check all values before proceeding */
	for (size_t i = 0; i < a.val.size(); i++) {
		assert(a.val[i] > 0. && a.val[i] < 1.);
		assert(b[i] == 0. || b[i] == 1.);
	}
#endif // !NDEBUG
#endif // TOO_SLOW

	index_t out = a.tape.rec< AggregBernoulliLogLikelihood >(a, { b.val });
	return Spy{ sum_log_dbern(Tensor(a), Tensor(b)), a.tape, out};
}