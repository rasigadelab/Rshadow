/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "op_base.h"

/****************************************************************************
*** AGGREGATORS WITH SCALAR OUTPUT
*****************************************************************************/

struct AggregSum : OpIn::Range, OpConst::None, OpOut::Scalar, HessianAlwaysZero {

	AggregSum(const OpIn::Range in, const OpConst::None constant,
		const OpOut::Scalar out) : OpIn::Range{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {			
		double y = 0.0;
		for(index_t i = in.begin(); i < in.end(); i++) {
			y += v[i];
		}
		v[out[0]] = y;
	}
	struct LocalDiff {
		const index_t n = 0;
		constexpr double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j < n);
			/* Grad[x + y, {x, y}] = 1 */
			return 1.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[x + y, {x, y}], {x, y}] = 0 */
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ index_t(in.size()) };
	}
};

struct AggregSumOfSquares : OpIn::Range, OpConst::None, OpOut::Scalar, HessianOffDiagAlwaysZero {

	AggregSumOfSquares(const OpIn::Range in, const OpConst::None constant,
		const OpOut::Scalar out) : OpIn::Range{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double y = 0.0;
		for (index_t i = in.begin(); i < in.end(); i++) {
			y += v[i] * v[i];
		}
		v[out[0]] = y;
	}
	struct LocalDiff {
		const AggregSumOfSquares& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j < n);
			/* Grad[x^2 + y^2, {x, y}] = {2 x,2 y}  */
			return 2. * v[op.in[j]];
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[x^2 + y^2, {x, y}], {x, y}]
			2 0
			0 2
			*/
			if (j != k) return 0.;
			else return 2.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(in.size()) };
	}
};

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct AggregDotProd;

template<> struct AggregDotProd< IN_FREE, IN_FREE > :
	OpIn::RangePair, OpConst::None, OpOut::Scalar, HessianDiagAlwaysZero, OpIsCommutable
{

	template< bool A_IS_FREE, bool B_IS_FREE> using family_t = AggregDotProd< A_IS_FREE, B_IS_FREE >;

	AggregDotProd(const OpIn::RangePair in, const OpConst::None constant, const OpOut::Scalar out) : OpIn::RangePair{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		assert(in.left.size() == in.right.size());
		assert((in.right.begin() >= in.left.end()) || (in.left.begin() >= in.right.end()));
		double y = 0.0;
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			y += v[in.left[i]] * v[in.right[i]];
		}
		v[out[0]] = y;
	}
	struct LocalDiff {
		const AggregDotProd& op;
		const std::vector< double >& v;
		const index_t n = 0;
		/* Assume vectorized { left, right } indexing */
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j < 2 * n);

			/* Grad[a*x + b*y + c*z, {a,b,c,x,y,z}] */
			/* {x,y,z,a,b,c} */	

			if (j < index_t(n)) {
				return v[op.in.right[j]];
			}
			else {
				return v[op.in.left[j - n]];
			}
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j < 2 * n);
			assert(k < 2 * n);
			/* Grad[Grad[a*x + b*y + c*z, {a,b,c,x,y,z}],{a,b,c,x,y,z}] 

			0	0	0	1	0	0
			0	0	0	0	1	0
			0	0	0	0	0	1
			1	0	0	0	0	0
			0	1	0	0	0	0
			0	0	1	0	0	0
			*/		

			if ((j + n == k) || (k + n == j)) {
				return 1.0;
			}
			else {
				return 0.0;
			}
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{*this, v, index_t(in.left.size()) };
	}
};

template<> struct AggregDotProd< IN_FREE, IN_FIXED > : OpIn::Range, OpConst::Vector, OpOut::Scalar, HessianAlwaysZero {

	template< bool A_IS_FREE, bool B_IS_FREE> using family_t = AggregDotProd< A_IS_FREE, B_IS_FREE >;

	AggregDotProd(const OpIn::Range in, const OpConst::Vector constant, const OpOut::Scalar out) : OpIn::Range{ in },
		OpConst::Vector{ constant }, OpOut::Scalar{ out }
	{
		assert(this->in.size() == this->constant.size());
	}

	void evaluate(std::vector< double >& v) const {
		assert(in.size() == constant.size());		
		double y = 0.0;
		for (index_t i = 0; i < index_t(in.size()); i++) {
			y += v[in[i]] * constant[i];
		}
		v[out[0]] = y;
	}
	struct LocalDiff {
		const AggregDotProd< IN_FREE, IN_FIXED >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		/* Assume vectorized { left, right } indexing */
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j < n);

			/* Grad[a*x + b*y + c*z, {a,b,c}] */
			/* {x,y,z} */
			return op.constant[j];
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[a*x + b*y + c*z, {a,b,c}],{a,b,c}] = { 0 }	*/
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(in.size()) };
	}
};

template<> struct AggregDotProd<IN_FIXED, IN_FREE> : AggregDotProd<IN_FREE, IN_FIXED> {
	using AggregDotProd<IN_FREE, IN_FIXED>::AggregDotProd;
};

/* Special operator, ll = sum(log( p * y + (1-p)*(1-y))) where p in a vector
of probabilities in ]0,1[ and y is a binary vector of observed events in {0,1} */
struct AggregBernoulliLogLikelihood : OpIn::Range, OpConst::Vector, OpOut::Scalar, HessianOffDiagAlwaysZero {

	AggregBernoulliLogLikelihood(const OpIn::Range in, const OpConst::Vector constant, const OpOut::Scalar out) : 
		OpIn::Range{ in }, OpConst::Vector{ constant }, OpOut::Scalar{ out }
	{
		assert(this->in.size() == this->constant.size());
	}

	void evaluate(std::vector< double >& v) const {
		assert(in.size() == constant.size());
		double y = 0.0;
		for (index_t i = 0; i < index_t(in.size()); i++) {
			assert(constant[i] == 0.0 || constant[i] == 1.0);
			if (constant[i]) {
				y += std::log(v[in[i]]);
			}
			else {
				y += std::log1p(-v[in[i]]);
			}			
		}
		v[out[0]] = y;
	}
	struct LocalDiff {
		const AggregBernoulliLogLikelihood& op;
		const std::vector< double >& v;
		const index_t n = 0;
		
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j < n);

			/* Grad[a*x + (1-a)*(1-x) + b*y + (1-b)*(1-y), {a,b}] =
			{2 x-1,2 y-1}  = either 1 or -1 without log.
			
			With the log,
			Grad[log(a*x + (1-a)*(1-x)) + log(b*y + (1-b)*(1-y)), {a,b}] =
			{(2 x-1)/((1-a) (1-x)+a x),(2 y-1)/((1-b) (1-y)+b y)}

			Grad[log(a*x + (1-a)*(1-x)) + log(b*y + (1-b)*(1-y)), {a,b}] where x=0, y=0
			{-(1/(1-a)),-(1/(1-b))}

			Grad[log(a*x + (1-a)*(1-x)) + log(b*y + (1-b)*(1-y)), {a,b}] where x=1, y=1
			{1/a,1/b}

			Hence, gradient is 1/a if x=1 or 1/(a-1) if x=0
			*/

			const double& a = v[op.in[j]];
			return op.constant[j] ? 1./a : 1./(a - 1.);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j < 2 * n);
			assert(k < 2 * n);
			/* 
			Grad[Grad[log(a*x + (1-a)*(1-x)) + log(b*y + (1-b)*(1-y)), {a,b}], {a,b}] where x=0, y=0
			-(1/(1-a)^2)	           0
					   0	-(1/(1-b)^2)

			Grad[Grad[log(a*x + (1-a)*(1-x)) + log(b*y + (1-b)*(1-y)), {a,b}], {a,b}] where x=1, y=1
			-(1/a^2)	       0
				   0	-(1/b^2)
			*/
			if (j != k) return 0.;
			const double& a = v[op.in[j]];
			if (op.constant[j]) {
				return -1. / (a * a);
			}
			else {
				const double am1 = a - 1.;
				return -1. / (am1 * am1);
			}
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(in.size()) };
	}
};

/****************************************************************************
*** OPERATOR DECLARATION
*****************************************************************************/

using op_aggregate_types = std::tuple< 
	AggregSum, AggregSumOfSquares, AggregBernoulliLogLikelihood,
	AggregDotProd< IN_FREE, IN_FREE >, AggregDotProd< IN_FREE, IN_FIXED >, AggregDotProd< IN_FIXED, IN_FREE >
>;