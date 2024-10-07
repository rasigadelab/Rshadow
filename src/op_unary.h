/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "op_base.h"
#include <cmath>
#include "rmath_bridge.h"

/****************************************************************************
*** IDENTITY
*****************************************************************************/

/* y = x */
struct IdentityScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar, PartialAlwaysOne, HessianAlwaysZero {

	IdentityScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}
		
	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = x;
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			return 1.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};

/* y = x */
struct IdentityVector : OpIn::Range, OpConst::None, OpOut::Range, 
	PartialAlwaysOne, HessianAlwaysZero, OpOutSize::Range_None_RangeSize
{

	IdentityVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(v.size() >= in.size() + out.size());
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			v[out[i]] = v[in[i]];
		}
	}
	struct LocalDiff {
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{a, b, c}, {a, b, c}] = Identity matrix*/
			if (j == i) return 1.;
			else return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {			
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{a, b, c}, {a, b, c}], {a, b, c}] */
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ index_t(out.size()) };
	}
};

/****************************************************************************
*** TRIVIAL, INTEGER CONSTANT
*****************************************************************************/

/* These are operators with self-cancelling partials or trivial results 
that may arise in some complex operations. We may chose to raise warnings. */

/* y _|_ x with constant integer result (zero or one).
Set RESULT = 1 for y = x/x or x^0
Set RESULT = 0 for y = x-x or x*0
*/
template< long RESULT >
struct TrivialScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar, PartialAlwaysZero, HessianAlwaysZero {

	TrivialScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		y = double(RESULT);
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};

template< long RESULT >
struct TrivialVector : OpIn::Range, OpConst::None, OpOut::Range, PartialAlwaysZero, HessianAlwaysZero, OpOutSize::Range_None_RangeSize {

	TrivialVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(v.size() >= in.size() + out.size());
		assert(in.size() == out.size());
		const double x = double(RESULT);
		for (index_t i = 0; i < index_t(in.size()); i++) {
			v[out[i]] = x;
		}
	}
	struct LocalDiff {
		const index_t n = 0;
		constexpr double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ index_t(out.size()) };
	}
};

/****************************************************************************
*** NEGATE, y = -x
*****************************************************************************/

/* y = -x */
struct NegateScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar, HessianAlwaysZero {

	NegateScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) : 
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}
		
	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = -x;
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			return -1.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};
/* y = -x */
struct NegateVector : OpIn::Range, OpConst::None, OpOut::Range, HessianAlwaysZero, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	NegateVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			v[out[i]] = -v[in[i]];
		}
	}
	struct LocalDiff {
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{-a, -b, -c}, {a, b, c}] = -Identity matrix*/
			if (j == i) return -1.;
			else return 0.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{-a, -b, -c}, {a, b, c}], {a, b, c}] */
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ index_t(out.size()) };
	}
};

/****************************************************************************
*** INVERT, y = 1/x
*****************************************************************************/

/* y = 1/x */
struct InvertScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	InvertScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = 1. / x;
	}
	struct LocalDiff {
		const double x_inv; // 1/x
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[{1/a}, {a}] */
			return -x_inv * x_inv;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{1/a}, {a}], {a}] */
			return 2.0 * x_inv * x_inv * x_inv;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ 1.0 / v[in[0]] };
	}
};
struct InvertVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	InvertVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			v[out[i]] = 1. / v[in[i]];
		}
	}
	struct LocalDiff {
		const InvertVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{1/a, 1/b}, {a, b}] =
			(-(1/a^2)		   0
					0	-(1/b^2) */
			if (j != i) return 0.;
			const double x_inv = 1. / v[op.in[i]];
			return -x_inv * x_inv;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{1/a, 1/b}, {a, b}], {a, b}] =
			{ 2/a^3, 0}	{ 0,     0}
			{     0, 0}	{ 0, 2/b^3} */
			if ((j != i) || (k != i)) return 0.;
			const double x_inv = 1. / v[op.in[i]];
			return 2.0 * x_inv * x_inv * x_inv;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** SQUARE, y = x*x
*****************************************************************************/

/* y = x^2 */
struct SquareScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	SquareScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = x * x;
	}
	struct LocalDiff {
		const double x;
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[{x^2}, {x}] = 2x */
			return 2. * x;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{x^2}, {x}], {x}] = 2 */
			return 2.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ v[in[0]] };
	}
};

struct SquareVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	SquareVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = x * x;
		}
	}
	struct LocalDiff {
		const SquareVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{x^2, y^2}, {x, y}] =
			2x	 0
			 0	2y */
			if (j != i) return 0.;
			else return 2. * v[op.in[i]];
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{x^2, y^2}, {x, y}], {x, y}] =
			{ 2, 0}	{ 0, 0}
			{ 0, 0}	{ 0, 2} */
			if ((j != i) || (k != i)) return 0.;			
			else return 2.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** CUBE, y = x*x*x
*****************************************************************************/

/* y = x^3 */
struct CubeScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	CubeScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = x * x * x;
	}
	struct LocalDiff {
		const double x;
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[{x^3}, {x}] = 3x� */
			return 3. * x * x;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{x^3}, {x}], {x}] = 6x */
			return 6. * x;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ v[in[0]] };
	}
};

struct CubeVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	CubeVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = x * x * x;
		}
	}
	struct LocalDiff {
		const CubeVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{x^3, y^3}, {x, y}] =
			3x�	  0
			  0	3y� */
			if (j != i) return 0.;
			const double& x = v[op.in[i]];
			return 3. * x * x;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{x^3, y^3}, {x, y}], {x, y}] =
			{ 6x, 0}	{ 0,  0}
			{  0, 0}	{ 0, 6y} */
			if ((j != i) || (k != i)) return 0.;
			const double& x = v[op.in[i]];
			return 6. * x;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** NATURAL LOGARITHM, y = log x
*****************************************************************************/

struct LogScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	LogScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = log(x);
	}
	struct LocalDiff {
		const double x_inv;
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[{Log(x)}, {x}] = 1/x */
			return x_inv;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{Log(x)}, {x}], {x}] = -1/x� */
			return -x_inv * x_inv;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ 1. / v[in[0]] };
	}
};

struct LogVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	LogVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = log(x);
		}
	}
	struct LocalDiff {
		const LogVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{Log(x), Log(y)}, {x, y}] =
			1/x	  0
			  0	1/y */
			if (j != i) return 0.;
			return 1. / v[op.in[i]];
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{Log(x), Log(y)}, {x, y}], {x, y}] =
			{ -1/x�, 0}	{ 0,     0}
			{     0, 0}	{ 0, -1/y�} */
			if ((j != i) || (k != i)) return 0.;
			const double& x_inv = 1. / v[op.in[i]];
			return -x_inv * x_inv;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** LOG1P, y = log(1 + x)
*****************************************************************************/

struct Log1pScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	Log1pScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = log1p(x);
	}
	struct LocalDiff {
		const double x_inv_1p; // 1 / (x + 1)
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[{Log1p(x)}, {x}] = 1/(x+1) */
			return x_inv_1p;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{Log1p(x)}, {x}], {x}] = -1/(x+1)� */
			return -x_inv_1p * x_inv_1p;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ 1. / (v[in[0]] + 1.) };
	}
};

struct Log1pVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	Log1pVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = log1p(x);
		}
	}
	struct LocalDiff {
		const Log1pVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{Log1p(x), Log1p(y)}, {x, y}] =
			1/(x+1)	        0
			      0	  1/(y+1) */
			if (j != i) return 0.;
			return 1. / (v[op.in[i]] + 1.);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{Log1p(x), Log1p(y)}, {x, y}], {x, y}] =
			{ -1/(x+1)�, 0}	{ 0,         0}
			{         0, 0}	{ 0, -1/(y+1)�} */
			if ((j != i) || (k != i)) return 0.;
			const double x_inv_1p = 1. / (v[op.in[i]] + 1.);
			return -x_inv_1p * x_inv_1p;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** LOG1M, y = log(1 - x)
*****************************************************************************/

struct Log1mScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	Log1mScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = log1p(-x);
	}
	struct LocalDiff {
		const double x_inv_1m; // 1 / (1 - x)
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[{Log1p(-x)}, {x}] = -1/(1-x) */
			return -x_inv_1m;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{Log1p(-x)}, {x}], {x}] = -1/(1-x)� */
			return -x_inv_1m * x_inv_1m;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ 1. / (1. - v[in[0]]) };
	}
};

struct Log1mVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	Log1mVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = log1p(-x);
		}
	}
	struct LocalDiff {
		const Log1mVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{Log1p(-x), Log1p(-y)}, {x, y}] =
			-1/(1-x)	        0
				  0	  -1/(1-y) */
			if (j != i) return 0.;
			return -1. / (1. - v[op.in[i]]);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{Log1p(-x), Log1p(-y)}, {x, y}], {x, y}] =
			{ -1/(1-x)�, 0}	{ 0,         0}
			{         0, 0}	{ 0, -1/(1-y)�} */
			if ((j != i) || (k != i)) return 0.;
			const double x_inv_1m = 1. / (1. - v[op.in[i]]);
			return -x_inv_1m * x_inv_1m;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** EXPONENTIATE, y = exp(x)
*****************************************************************************/

struct ExpScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	ExpScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = exp(x);
	}
	struct LocalDiff {
		const double x_exp; // e^x
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[{Exp(x)}, {x}] = Exp(x) */
			return x_exp;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{Exp(x)}, {x}], {x}] = Exp(x) */
			return x_exp;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ exp(v[in[0]]) };
	}
};

struct ExpVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	ExpVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = exp(x);
		}
	}
	struct LocalDiff {
		const ExpVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{Exp(x), Exp(y)}, {x, y}] =
			e^x	    0
			  0	  e^y */
			if (j != i) return 0.;
			return exp(v[op.in[i]]);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{Exp(x), Exp(y)}, {x, y}], {x, y}] =
			{ e^x, 0}	{ 0,   0}
			{   0, 0}	{ 0, e^y} */
			if ((j != i) || (k != i)) return 0.;
			return exp(v[op.in[i]]);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** SELF-POWER, y = x^x
*****************************************************************************/

struct SelfPowerScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	SelfPowerScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = pow(x, x);
	}
	struct LocalDiff {
		const double x;
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[{x^x}, {x}] = {x^x (log(x)+1)}  */
			return pow(x, x) * (1. + log(x));
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{x^x}, {x}], {x}] = (x^(x-1)+x^x (log(x)+1)^2)  */
			const double lx1p = log(x) + 1.;
			return pow(x, x - 1.) + pow(x, x) * lx1p * lx1p;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ v[in[0]] };
	}
};

struct SelfPowerVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	SelfPowerVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = pow(x, x);
		}
	}
	struct LocalDiff {
		const SelfPowerVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{x^x, y^y}, {x, y}] */
			if (j != i) return 0.;
			const double& x = v[op.in[i]];
			return pow(x, x) * (1. + log(x));
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{x^x, y^y}, {x, y}], {x, y}] */
			if ((j != i) || (k != i)) return 0.;
			const double& x = v[op.in[i]];
			const double lx1p = log(x) + 1.;
			return pow(x, x - 1.) + pow(x, x) * lx1p * lx1p;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** LOG-GAMMA FUNCTION
****************************************************************************/

struct LogGammaScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	LogGammaScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = std::lgamma(x);
	}
	struct LocalDiff {
		double x;
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[LogGamma[x], {x}] = PolyGamma[0, x] = digamma(x)  */
			return rmath_digamma(x);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[LogGamma[x], {x}], {x}] = PolyGamma[1, x] = trigamma(x)  */
			return rmath_trigamma(x);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ v[in[0]] };
	}
};

struct LogGammaVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	LogGammaVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			v[out[i]] = std::lgamma(v[in[i]]);
		}
	}
	struct LocalDiff {
		const LogGammaVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{LogGamma[x], LogGamma[y]}, {x, y}] */
			if (j != i) return 0.;
			const double& x = v[op.in[i]];
			return rmath_digamma(x);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{LogGamma[x], LogGamma[y]}, {x, y}], {x, y}] */
			if ((j != i) || (k != i)) return 0.;
			const double& x = v[op.in[i]];
			return rmath_trigamma(x);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** LOGIT FUNCTION y = log(x / (1-x))
*****************************************************************************/

struct LogitScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	LogitScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = std::log(x / (1. - x));
	}
	struct LocalDiff {
		const double x;
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* 1/(x-x^2) */
			return 1. / (x - x * x);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* 1/(x-1)^2-1/x^2 */
			const double xm1_inv = 1. / (x - 1.);
			return xm1_inv * xm1_inv - 1. / (x * x);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ v[in[0]] };
	}
};

struct LogitVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	LogitVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = std::log(x / (1. - x));
		}
	}
	struct LocalDiff {
		const LogitVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{Logit[x], Logit[y]}, {x, y}] */
			if (j != i) return 0.;
			const double& x = v[op.in[i]];
			/* 1/(x-x^2) */
			return 1. / (x - x * x);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{Logit[x], Logit[y]}, {x, y}], {x, y}] */
			if ((j != i) || (k != i)) return 0.;
			const double& x = v[op.in[i]];
			/* 1/(x-1)^2-1/x^2 */
			const double xm1_inv = 1. / (x - 1.);
			return xm1_inv * xm1_inv - 1. / (x * x);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** LOGISTIC FUNCTION y = 1/(1 + e^-x)
*****************************************************************************/

struct LogisticScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	LogisticScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = 1./(1. + exp(-x));
	}
	struct LocalDiff {
		double exp_mx = 0.0; // exp(-x)
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[1/(1+E^-x), {x} ] = E^-x/(E^-x+1)^2 */
			const double exp_mx_p1 = exp_mx + 1.;
			return exp_mx / (exp_mx_p1 * exp_mx_p1);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[1/(1+E^-x), {x} ], {x} ]
			
			-((E^x (E^x-1))/(E^x+1)^3) , beware of the use of E^x
			instead of E^-x, resulting in a change of sign */
			const double exp_mx_p1 = exp_mx + 1.;
			return exp_mx * (exp_mx - 1.) / (exp_mx_p1 * exp_mx_p1 * exp_mx_p1);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ exp(-v[in[0]]) };
	}
};

struct LogisticVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	LogisticVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = 1. / (1. + exp(-x));
		}
	}
	struct LocalDiff {
		const LogisticVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			if (j != i) return 0.;
			/* E^-x/(E^-x+1)^2 */
			const double exp_mx = exp(-v[op.in[i]]);
			const double exp_mx_p1 = exp_mx + 1.;
			return exp_mx / (exp_mx_p1 * exp_mx_p1);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			if ((j != i) || (k != i)) return 0.;
			const double exp_mx = exp(-v[op.in[i]]);
			/* -((E^x (E^x-1))/(E^x+1)^3) , beware of the use of E^x
			instead of E^-x, resulting in a change of sign */
			const double exp_mx_p1 = exp_mx + 1.;
			return exp_mx * (exp_mx - 1.) / (exp_mx_p1 * exp_mx_p1 * exp_mx_p1);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** COSINE, y = cos(x)
*****************************************************************************/

struct CosScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	CosScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = cos(x);
	}
	struct LocalDiff {
		const double x;
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[{cos(x)}, {x}] = -sin(x) */
			return -sin(x);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{cos(x)}, {x}], {x}] = -cos(x) */
			return -cos(x);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ v[in[0]] };
	}
};

struct CosVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	CosVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = cos(x);
		}
	}
	struct LocalDiff {
		const CosVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{cos(x)}, {x}] = -sin(x) */
			if (j != i) return 0.;
			return -sin(v[op.in[i]]);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{cos(x)}, {x}], {x}] = -cos(x) */
			if ((j != i) || (k != i)) return 0.;
			return -cos(v[op.in[i]]);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** SINE, y = sin(x)
*****************************************************************************/

struct SinScalar : OpIn::Array<1>, OpConst::None, OpOut::Scalar {

	SinScalar(const OpIn::Array<1> in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		double& y = v[out[0]];
		double& x = v[in[0]];
		y = sin(x);
	}
	struct LocalDiff {
		const double x;
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[{sin(x)}, {x}] = cos(x) */
			return cos(x);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{sin(x)}, {x}], {x}] = -sin(x) */
			return -sin(x);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ v[in[0]] };
	}
};

struct SinVector : OpIn::Range, OpConst::None, OpOut::Range, OpIsElementWise, OpOutSize::Range_None_RangeSize {

	SinVector(const OpIn::Range in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			const double& x = v[in[i]];
			v[out[i]] = sin(x);
		}
	}
	struct LocalDiff {
		const SinVector& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{sin(x)}, {x}] = cos(x) */
			if (j != i) return 0.;
			return cos(v[op.in[i]]);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{sin(x)}, {x}], {x}] = -sin(x) */
			if ((j != i) || (k != i)) return 0.;
			return -sin(v[op.in[i]]);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(out.size()) };
	}
};

/****************************************************************************
*** OPERATOR DECLARATION
*****************************************************************************/

using op_unary_types = std::tuple<
	IdentityScalar, IdentityVector,
	TrivialScalar<0>, TrivialVector<0>,
	TrivialScalar<1>, TrivialVector<1>,
	NegateScalar, NegateVector,
	InvertScalar, InvertVector,
	SquareScalar, SquareVector,
	CubeScalar, CubeVector,
	LogScalar, LogVector,
	Log1pScalar, Log1pVector,
	Log1mScalar, Log1mVector,
	ExpScalar, ExpVector,
	SelfPowerScalar, SelfPowerVector,
	LogGammaScalar, LogGammaVector,
	LogitScalar, LogitVector,
	LogisticScalar, LogisticVector,
	CosScalar, CosVector,
	SinScalar, SinVector
>;