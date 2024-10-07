/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "op_base.h"

/****************************************************************************
*** DIVISION, SCALAR DIVIDED BY SCALAR
*****************************************************************************/

/* y = a / b */
template< bool A_IS_FREE = IN_FREE, bool B_IS_FREE = IN_FREE > struct DivideScalarScalar;

/* y = a / b; a is free, b is free */
template<> struct DivideScalarScalar<IN_FREE, IN_FREE> :
	OpIn::ScalarScalar, OpConst::None, OpOut::Scalar
{
	/* Used by tape recompilation code */
	template< bool a, bool b> using family_t = DivideScalarScalar< a, b >;

	DivideScalarScalar(const OpIn::ScalarScalar in, const OpConst::None constant, const OpOut::Scalar out) : 
		OpIn::ScalarScalar{ in }, OpOut::Scalar{ out } 
	{
		assert((this->in.left[0] != this->in.right[0]) && "Duplicate edge detected. Please use Trivial<1> operator instead.");
	}

	/* Forward pass, compute function value only */
	void evaluate(std::vector< double >& v) const {
		v[out[0]] = v[in.left[0]] / v[in.right[0]];
	}
	struct LocalDiff {

		const double a;
		const double bi; /* Inverse of b */

		double partial(const index_t i, const index_t j) const {
			/* Grad[a/b, {a,b}] = {1/b,-(a/b^2)} */
			assert(i == 0);
			assert(j == 0 || j == 1);
			if (j == 0) return bi;
			else return -a * bi * bi;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[a/b, {a,b}], {a,b}] 
			0			-(1/b^2)
			-(1/b^2)	(2 a)/b^3
			*/
			assert(i == 0);
			assert(j == 0 || j == 1);
			assert(k == 0 || k == 1);

			if (j == 0 && k == 0) return 0.0;
			else if (k == 1 - j) return -bi * bi;
			else return 2. * a * bi * bi * bi;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ v[in.left[0]], 1.0 / v[in.right[0]] };
	}
};


/* y = a / b; a is free, b is fixed */
template<> struct DivideScalarScalar<IN_FREE, IN_FIXED> : OpIn::Array<1>, OpConst::Array<1>, OpOut::Scalar {

	template< bool a, bool b> using family_t = DivideScalarScalar< a, b >;

	/* We may store 1/constant instead but this would hurt readability, so we refrain... */
	DivideScalarScalar(const OpIn::Array<1> in, const OpConst::Array<1> constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpConst::Array<1>{ constant }, OpOut::Scalar{ out } {}

	/* Forward pass, compute function value only */
	void evaluate(std::vector< double >& v) const {
		v[out[0]] = v[in[0]] / constant[0];
	}
	struct LocalDiff {
		const double bi; /* Inverse of b */
		double partial(const index_t i, const index_t j) const {
			/* Grad[a/b, {a}] = { 1/b } */
			assert(i == 0);
			assert(j == 0);
			return bi;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ 1. / constant[0] };
	}
};

/* y = a / b; a is fixed, b is free */
template<> struct DivideScalarScalar<IN_FIXED, IN_FREE> :
	OpIn::Array<1>, OpConst::Array<1>, OpOut::Scalar
{

	template< bool a, bool b> using family_t = DivideScalarScalar< a, b >;

	DivideScalarScalar(const OpIn::Array<1> in, const OpConst::Array<1> constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out }, OpConst::Array<1>{ constant } {}

	/* Forward pass, compute function value only */
	void evaluate(std::vector< double >& v) const {
		v[out[0]] = constant[0] / v[in[0]];
	}
	struct LocalDiff {

		const double a;
		const double bi; /* Inverse of b */

		double partial(const index_t i, const index_t j) const {
			/* Grad[a/b, {b}] = {-(a/b^2)} */
			assert(i == 0);
			assert(j == 0);
			return -a * bi * bi;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[a/b, {b}], {b}] = (2 a)/b^3 */
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			return 2. * a * bi * bi * bi;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ constant[0], 1. / v[in[0]] };
	}
};

/****************************************************************************
*** DIVISION, VECTOR DIVIDED BY SCALAR
*****************************************************************************/

template< bool A_IS_FREE = IN_FREE, bool B_IS_FREE = IN_FREE > struct DivideVectorScalar;

template<> struct DivideVectorScalar< IN_FREE, IN_FREE > : 
	OpIn::RangeArrayPair<1>, OpConst::None, OpOut::Range, 
	OpIsElementWise, OpOutSize::RangeArray_None_RangeSize {

	template< bool a, bool b> using family_t = DivideVectorScalar< a, b >;

	DivideVectorScalar(const OpIn::RangeArrayPair<1> in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::RangeArrayPair<1>{ in }, OpOut::Range{ out }
	{
		assert((this->in.left.end() <= this->in.right[0]) || (this->in.right[0] < this->in.left.begin()));
	}
	void evaluate(std::vector< double >& v) const {
		const double& b = v[in.right[0]];
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in.left[i]];
			y = a / b;
		}
	}
	struct LocalDiff {
		const DivideVectorScalar< IN_FREE, IN_FREE >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{a/x, b/x}, {a, b, x}] 
					1/x	  0	  -(a/x^2)
					0	1/x	  -(b/x^2) */
			if (j == i) return 1.0 / v[op.in.right[0]]; // 1/x
			if (j == n) return -v[op.in.left[i]] / (v[op.in.right[0]] * v[op.in.right[0]]); // -(a or b) / (x*x)
			else return 0.0;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[{a/x, b/x}, {a, b, x}], {a, b, x}]

			{0,0,-(1/x^2)}	{0,0,       0}	{-(1/x^2),       0,(2 a)/x^3}
			{0,0,       0}	{0,0,-(1/x^2)}	{       0,-(1/x^2),(2 b)/x^3}		*/
			if (  (j == i && k == n) ||
				  (j == n && k == i)  )
			{
				return -1.0 / (v[op.in.right[0]] * v[op.in.right[0]]); //-(1/x^2)
			}
			if (j == n && k == n) {
				return 2.0 * v[op.in.left[i]] / (v[op.in.right[0]] * v[op.in.right[0]] * v[op.in.right[0]]); //2 (a or b)/x^3
			}
			else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(in.left.size()) };
	}
};

template<> struct DivideVectorScalar< IN_FREE, IN_FIXED > : 
	OpIn::Range, OpConst::Array<1>, OpOut::Range, 
	OpIsElementWise, OpOutSize::Range_Scalar_RangeSize {

	template< bool a, bool b> using family_t = DivideVectorScalar< a, b >;

	DivideVectorScalar(const OpIn::Range in, const OpConst::Array<1> constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Array<1>{ constant }, OpOut::Range{ out } {}

	void evaluate(std::vector< double >& v) const {
		const double& b = constant[0];
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in[i]];
			y = a / b;
		}
	}
	struct LocalDiff {

		const double ci = 0.0; // The inverse scalar constant
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			/* Grad[{a/c, b/c}, {a, b}]
			1/c	  0
			0	1/c
			*/
			assert(i < n);
			assert(j < n);
			if (j == i) return ci;
			else return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{a/c, b/c}, {a, b}], {a, b}]  = { 0 }*/
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ 1. / constant[0], index_t(in.size()) };
	}
};

template<> struct DivideVectorScalar< IN_FIXED, IN_FREE > : 
	OpIn::Array<1>, OpConst::Vector, OpOut::Range,
	OpIsElementWise, OpOutSize::Scalar_Vector_VectorSize  
{

	template< bool a, bool b> using family_t = DivideVectorScalar< a, b >;

	DivideVectorScalar(const OpIn::Array<1> in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Array<1>{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {}

	void evaluate(std::vector< double >& v) const {
		const double bi = 1. / v[in[0]];
		for (index_t i = 0; i < index_t(constant.size()); i++) {
			double& y = v[out[i]];
			y = constant[i] * bi;
		}
	}
	struct LocalDiff {
		const DivideVectorScalar< IN_FIXED, IN_FREE >& op;
		const double xi; // Inverse of scalar right argument
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{c1/x, c2/x}, {x}] = { -c1 / x², -c2 / x² } */
			assert(i < n);
			assert(j == 0);			
			return -op.constant[i] * xi * xi;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{c1/x, c2/x}, {x}], {x}] = {2 c1/x^3, 2 c2/x^3} */
			return 2. * op.constant[i] * xi * xi * xi;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, 1. / v[in[0]], index_t(constant.size()) };
	}

};

/****************************************************************************
*** DIVISION, SCALAR DIVIDED BY VECTOR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct DivideScalarVector;

template<> struct DivideScalarVector< IN_FREE, IN_FREE > : 
	OpIn::ArrayRangePair<1>, OpConst::None, OpOut::Range,
	OpIsElementWise, OpOutSize::ArrayRange_None_RangeSize 
{

	template< bool a, bool b> using family_t = DivideScalarVector< a, b >;

	DivideScalarVector(const OpIn::ArrayRangePair<1> in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::ArrayRangePair<1>{ in }, OpOut::Range{ out }
	{
		assert(this->in.right.size() == this->out.size());
		assert((this->in.left[0] < this->in.right.begin()) || (this->in.right.end() <= this->in.left[0]));
	}
	void evaluate(std::vector< double >& v) const {
		const double& a = v[in.left[0]];
		for (index_t i = 0; i < index_t(in.right.size()); i++) {
			double& y = v[out[i]];
			const double& b = v[in.right[i]];
			y = a / b;
		}
	}

	struct LocalDiff {
		const DivideScalarVector< IN_FREE, IN_FREE >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{x/a, x/b}, {x, a, b}]
				1/a	-x/a^2	     0
				1/b	     0	-x/b^2	*/
			if (j == 0) return 1.0 / v[op.in.right[i]];
			if (j == 1 + i) return -v[op.in.left[0]] / (v[op.in.right[i]] * v[op.in.right[i]]);
			else return 0.0;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[{x/a, x/b}, {x, a, b}], {x, a, b}]
			 
				{0, -1/a^2,        0}	{ -1/a^2, 2 x/a^3, 0}	{     0, 0,        0}
				{0,        0, -1/b^2}	{      0,       0, 0}	{-1/b^2, 0, 2 x /b^3}
						*/
			if ((j == 0 && k == 1 + i) ||
				(j == 1 + i && k == 0))
			{
				const double& right_inv = 1. / v[op.in.right[i]];
				return -right_inv * right_inv;
			}
			if (j == 1 + i && k == 1 + i) {
				const double& right_inv = 1. / v[op.in.right[i]];
				return 2. * v[op.in.left[0]] * right_inv * right_inv * right_inv;
			}
			else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(in.left.size()) };
	}

};

template<> struct DivideScalarVector< IN_FREE, IN_FIXED > : 
	OpIn::Array<1>, OpConst::Vector, OpOut::Range,
	OpIsElementWise, OpOutSize::Scalar_Vector_VectorSize 
{

	template< bool a, bool b> using family_t = DivideScalarVector< a, b >;

	DivideScalarVector(const OpIn::Array<1> in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Array<1>{ in }, OpConst::Vector{ constant }, OpOut::Range{ out }
	{
		assert(this->constant.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {

		const double& a = v[in[0]];

		for (index_t i = 0; i < index_t(constant.size()); i++) {
			double& y = v[out[i]];
			const double& b = constant[i];
			y = a / b;
		}
	}
	struct LocalDiff {
		const DivideScalarVector< IN_FREE, IN_FIXED >& op;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{x/a, x/b}, {x}] = {1/a, 1/b} */
			assert(i < n);
			assert(j == 0);
			return 1. / op.constant[i];
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{x/a, x/b}, {x}], {x}] = 0 */
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, index_t(constant.size()) };
	}
};

template<> struct DivideScalarVector< IN_FIXED, IN_FREE > : 
	OpIn::Range, OpConst::Array<1>, OpOut::Range,
	OpIsElementWise, OpOutSize::Range_Scalar_RangeSize 
{

	template< bool a, bool b> using family_t = DivideScalarVector< a, b >;

	DivideScalarVector(const OpIn::Range in, const OpConst::Array<1> constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Array<1>{ constant }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = constant[0];
			const double& b = v[in[i]];
			y = a / b;
		}
	}
	struct LocalDiff {
		const DivideScalarVector< IN_FIXED, IN_FREE >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{x/a, x/b}, {a, b}]
				-x/a^2	     0
				     0	-x/b^2	*/			
			if (j == i) return -op.constant[0] / (v[op.in[i]] * v[op.in[i]]);
			else return 0.0;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[{x/a, x/b}, {a,b}], {a,b}]

				{ 2 x/a^3, 0}	{ 0,        0}
				{       0, 0}	{ 0, 2 x /b^3}
						*/
			if (j == i && k == i)
			{
				const double& right_inv = 1. / v[op.in[i]];
				return 2. * op.constant[0] * right_inv * right_inv * right_inv;
			}
			else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(constant.size()) };
	}
};

/****************************************************************************
*** DIVISION, VECTOR DIVIDED BY VECTOR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct DivideVectorVector;

template <> struct DivideVectorVector< IN_FREE, IN_FREE > : 
	OpIn::RangePair, OpConst::None, OpOut::Range,
	OpIsElementWise, OpOutSize::RangePair_None_RangeSize 
{

	template< bool a, bool b> using family_t = DivideVectorVector< a, b >;

	DivideVectorVector(const OpIn::RangePair in, const OpConst::None constant, const OpOut::Range out) : OpIn::RangePair{ in }, OpOut::Range{ out } {
		assert(this->in.left.size() == this->in.right.size());
		assert((this->in.right.begin() >= this->in.left.end()) || (this->in.left.begin() >= this->in.right.end()));
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.left.size() == in.right.size());
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in.left[i]];
			const double& b = v[in.right[i]];
			y = a / b;
		}
	}
	struct LocalDiff {
		const DivideVectorVector< IN_FREE, IN_FREE >& op;
		const std::vector< double >& v;
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {

			/* Grad[{a/x, b/y, c/z}, {a, b, c, x, y, z}] 
			1/x	0	0	-(a/x^2)	0			0
			0	1/y	0	0			-(b/y^2)	0
			0	0	1/z	0			0			-(c/z^2)
			*/

			if (j == i) {
				/* Partial wrt left element is right element */
				return 1.0 / v[op.in.right[i]];
			}
			if (j == i + n) {
				/* Partial wrt right element is left element */
				return -v[op.in.left[i]] / (v[op.in.right[i]] * v[op.in.right[i]]);
			}
			return 0.0;
		}

		double partial(const index_t i, const index_t j, const index_t k) {
			/*
			Grad[Grad[{a/x, b/y}, {a, b, x, y}], {a, b, x, y}]

				{0,0,-(1/x^2),0}	{0,0,0,		  0}	{-(1/x^2),0,(2 a)/x^3,0}	{0,		  0,0,		  0}
				{0,0,		0,0}	{0,0,0,-(1/y^2)}	{		0,0,		0,0}	{0,-(1/y^2),0,(2 b)/y^3}

			*/

			if (
				((j == i) && (k == i + n)) ||
				((j == i + n) && (k == i))
				) {
				return -1.0 / (v[op.in.right[i]] * v[op.in.right[i]]);
			}
			if ((j == i + n) && (k == i + n)) {
				return  2.0 * v[op.in.left[i]] / (v[op.in.right[i]] * v[op.in.right[i]] * v[op.in.right[i]]);
			}
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		assert(in.left.size() == in.right.size());
		return { *this, v, index_t(in.left.size()) };
	}
};

template<> struct DivideVectorVector< IN_FREE, IN_FIXED > : 
	OpIn::Range, OpConst::Vector, OpOut::Range, 
	OpIsElementWise, OpOutSize::Range_Vector_RangeSize
{

	template< bool a, bool b> using family_t = DivideVectorVector< a, b >;

	DivideVectorVector(const OpIn::Range in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {
		assert(this->in.size() == this->constant.size());
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in[i]];
			const double& b = constant[i];
			y = a / b;
		}
	}
	struct LocalDiff {
		const DivideVectorVector< IN_FREE, IN_FIXED >& op;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{a/x, b/y}, {a, b}] 
			1/x    0
			  0  1/y
			*/
			assert(i < n);
			assert(j < n);
			if (j == i) return 1. / op.constant[i];
			else return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[{a/x, b/y}, {a, b}], {a, b}] = { 0... }*/
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, index_t(constant.size()) };
	}
};

template<> struct DivideVectorVector< IN_FIXED, IN_FREE > : 
	OpIn::Range, OpConst::Vector, OpOut::Range,
	OpIsElementWise, OpOutSize::Range_Vector_RangeSize
{

	template< bool a, bool b> using family_t = DivideVectorVector< a, b >;

	DivideVectorVector(const OpIn::Range in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {
		assert(this->in.size() == this->constant.size());
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = constant[i];
			const double& b = v[in[i]];
			y = a / b;
		}
	}
	struct LocalDiff {
		const DivideVectorVector< IN_FIXED, IN_FREE >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{a/x, b/y}, {x, y}]
				-a/x^2	     0
					 0  -b/y^2  */
			assert(i < n);
			assert(j < n);
			if (j == i) return -op.constant[i] / (v[op.in[i]] * v[op.in[i]]);
			else return 0.0;
		}
		 double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[{a/x, b/y}, {x, y}], {x, y}] 
			
				{ 2 a/x^3, 0}	{0,		  0}
				{		0, 0}	{0, 2 b/y^3}
			
			*/
			 if (j == i && k == i)
			 {
				 const double& right_inv = 1. / v[op.in[i]];
				 return 2. * op.constant[i] * right_inv * right_inv * right_inv;
			 }
			 else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(constant.size()) };
	}
};

/****************************************************************************
*** OPERATOR DECLARATION
*****************************************************************************/

using op_divide_types = std::tuple<
	DivideScalarScalar< IN_FREE, IN_FREE >, DivideScalarScalar< IN_FREE, IN_FIXED >, DivideScalarScalar< IN_FIXED, IN_FREE >,
	DivideVectorScalar< IN_FREE, IN_FREE >, DivideVectorScalar< IN_FREE, IN_FIXED >, DivideVectorScalar< IN_FIXED, IN_FREE >,
	DivideScalarVector< IN_FREE, IN_FREE >, DivideScalarVector< IN_FREE, IN_FIXED >, DivideScalarVector< IN_FIXED, IN_FREE >,
	DivideVectorVector< IN_FREE, IN_FREE >, DivideVectorVector< IN_FREE, IN_FIXED >, DivideVectorVector< IN_FIXED, IN_FREE >
>;