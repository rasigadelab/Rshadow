/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "op_base.h"

/****************************************************************************
*** POWER, SCALAR RAISED TO SCALAR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct PowerScalarScalar;

template<> struct PowerScalarScalar<true, true> :
	OpIn::ScalarScalar, OpConst::None, OpOut::Scalar
{
	/* Used by tape recompilation code */
	template< bool a, bool b> using family_t = PowerScalarScalar< a, b >;

	PowerScalarScalar(const OpIn::ScalarScalar in, const OpConst::None constant, const OpOut::Scalar out) :
		OpIn::ScalarScalar{ in }, OpOut::Scalar{ out }
	{
		assert((this->in.left[0] != this->in.right[0]) && "Duplicate edge detected. Please use SelfPower operator instead.");
	}

	/* Forward pass, compute function value only */
	void evaluate(std::vector< double >& v) const {
		v[out[0]] = pow(v[in.left[0]], v[in.right[0]]);
	}
	struct LocalDiff {

		const double a;
		const double b;
		const double log_a;
		const double pow_a_bm1; // a ^ (b - 1)

		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0 || j == 1);
			/* Grad[a^b, {a,b}] */
			/* {b a^(b-1),a^b log(a)}  */
			if (j == 0) {
				return b * pow_a_bm1;
			}
			else return pow(a, b) * log_a;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0 || j == 1);
			assert(k == 0 || k == 1);
			/* Grad[Grad[a^b, {a,b}], {a,b}]
			
			(b-1) b a^(b-2)				a^(b-1)+b a^(b-1) log(a)
			a^(b-1)+b a^(b-1) log(a)	a^b log^2(a)
			*/

			if (j == k) {
				if (j == 0) return (b - 1.) * b * pow(a, b - 2.);
				return pow(a, b) * log_a * log_a;
			}
			return pow_a_bm1 + b * pow_a_bm1 * log_a;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		const double& a = v[in.left[0]], b = v[in.right[0]];
		return LocalDiff{ a, b, log(a), pow(a, b - 1.) };
	}
};

template<> struct PowerScalarScalar<true, false> : OpIn::Array<1>, OpConst::Array<1>, OpOut::Scalar {

	template< bool a, bool b> using family_t = PowerScalarScalar< a, b >;

	PowerScalarScalar(const OpIn::Array<1> in, const OpConst::Array<1> constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpConst::Array<1>{ constant }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		v[out[0]] = pow(v[in[0]], constant[0]);
	}

	struct LocalDiff {
		const double a;
		const double b;
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[a^b, {a}] = { b a^(b-1) } */
			return b * pow(a, b - 1.);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[a^b, {a}], {a}] = (b-1) b a^(b-2) */
			return (b - 1.) * b * pow(a, b - 2.);
		}
	};

	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ v[in[0]], constant[0] };
	}
};

template<> struct PowerScalarScalar<false, true> :
	OpIn::Array<1>, OpConst::Array<1>, OpOut::Scalar
{

	template< bool a, bool b> using family_t = PowerScalarScalar< a, b >;

	PowerScalarScalar(const OpIn::Array<1> in, const OpConst::Array<1> constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out }, OpConst::Array<1>{ constant } {}

	void evaluate(std::vector< double >& v) const {
		v[out[0]] = pow(constant[0], v[in[0]]);
	}

	struct LocalDiff {

		const double pow_ab; // a^b
		const double log_a; /* Inverse of b */

		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			/* Grad[a^b, {b}] = { a^b log(a) } */
			return pow_ab * log_a;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[a^b, {b}], {b}] = a^b log^2(a) */
			return pow_ab * log_a * log_a;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		const double& a = constant[0], b = v[in[0]];
		return LocalDiff{ pow(a,b), log(a) };
	}
};

/****************************************************************************
*** POWER, VECTOR RAISED TO SCALAR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct PowerVectorScalar;

template<> struct PowerVectorScalar< true, true > : 
	OpIn::RangeArrayPair<1>, OpConst::None, OpOut::Range, 
	OpIsElementWise, OpOutSize::RangeArray_None_RangeSize
{

	template< bool a, bool b> using family_t = PowerVectorScalar< a, b >;

	PowerVectorScalar(const OpIn::RangeArrayPair<1> in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::RangeArrayPair<1>{ in }, OpOut::Range{ out }
	{
		assert((this->in.left.end() <= this->in.right[0]) || (this->in.right[0] < this->in.left.begin()));
	}
	void evaluate(std::vector< double >& v) const {
		const double& b = v[in.right[0]];
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in.left[i]];
			y = pow(a, b);
		}
	}
	struct LocalDiff {
		const PowerVectorScalar< true, true >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n + 1);
			/* Grad[{a^x, b^x}, {a, b, x}]
					x a^(x-1)	       0	  a^x log a
					       0   x b^(x-1)      b^x log b */
			if (j == i) return v[op.in.right[0]] * pow(v[op.in.left[i]], v[op.in.right[0]] - 1.);
			if (j == n) return pow(v[op.in.left[i]], v[op.in.right[0]]) * log(v[op.in.left[i]]);
			else return 0.0;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n + 1);
			assert(k < n + 1);
			/* Grad[Grad[{a^x, b^x}, {a, b, x}], {a, b, x}]

			{(x-1) x a^(x-2), 0,a^(x-1)+x a^(x-1) log(a)}	{0,               0,                        0}	{ a^(x-1)+x a^(x-1) log(a),                        0, a^x log^2(a)}
			{              0, 0,                       0}	{0, (x-1) x b^(x-2), b^(x-1)+x b^(x-1) log(b)}	{                        0, b^(x-1)+x b^(x-1) log(b), b^x log^2(b)}	*/
			if (j == i && k == i) 
			{
				/* (x-1) x a^(x-2) */
				const double& a = v[op.in.left[i]];
				const double& x = v[op.in.right[0]];
				return (x - 1.) * x * pow(a, x - 2.);
			}
			if ((j == i && k == n) || (j == n && k == i))
			{
				/* a^(x-1)+x a^(x-1) log(a) */
				const double& a = v[op.in.left[i]];
				const double& x = v[op.in.right[0]];
				const double pow_a_xm1 = pow(a, x - 1.);
				return pow_a_xm1 + x * pow_a_xm1 * log(a);
			}
			if (j == n && k == n) {
				/* a^x log^2(a) */
				const double& a = v[op.in.left[i]];
				const double& x = v[op.in.right[0]];
				const double log_a = log(a);
				return pow(a, x) * log_a * log_a;
			}
			else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(in.left.size()) };
	}
};

template<> struct PowerVectorScalar< true, false > : 
	OpIn::Range, OpConst::Array<1>, OpOut::Range, 
	OpIsElementWise, OpOutSize::Range_Scalar_RangeSize
{

	template< bool a, bool b> using family_t = PowerVectorScalar< a, b >;

	PowerVectorScalar(const OpIn::Range in, const OpConst::Array<1> constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Array<1>{ constant }, OpOut::Range{ out } {}

	void evaluate(std::vector< double >& v) const {
		const double& b = constant[0];
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in[i]];
			y = pow(a, b);
		}
	}
	struct LocalDiff {
		const PowerVectorScalar< true, false >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{a^x, b^x}, {a, b}]
					x a^(x-1)	       0
						   0   x b^(x-1) */
			if (j != i) return 0.0;
			const double& a = v[op.in[i]];
			const double& x = op.constant[0];
			return x * pow(a, x - 1.);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{a^x, b^x}, {a, b}], {a, b}]

			{(x-1) x a^(x-2), 0}	{0,               0}
			{              0, 0}	{0, (x-1) x b^(x-2)}	*/
			if (j == i && k == i)
			{
				/* (x-1) x a^(x-2) */
				const double& a = v[op.in[i]];
				const double& x = op.constant[0];
				return (x - 1.) * x * pow(a, x - 2.);
			}
			else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(in.size()) };
	}
};

template<> struct PowerVectorScalar< false, true > : 
	OpIn::Array<1>, OpConst::Vector, OpOut::Range, 
	OpIsElementWise, OpOutSize::Scalar_Vector_VectorSize
{

	template< bool a, bool b> using family_t = PowerVectorScalar< a, b >;

	PowerVectorScalar(const OpIn::Array<1> in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Array<1>{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {}

	void evaluate(std::vector< double >& v) const {
		const double b = v[in[0]];
		for (index_t i = 0; i < index_t(constant.size()); i++) {
			double& y = v[out[i]];
			y = pow(constant[i], b);
		}
	}
	struct LocalDiff {
		const PowerVectorScalar< false, true >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j == 0);
			/* Grad[{a^x, b^x}, {x}]
					a^x log a
					b^x log b */
			const double& a = op.constant[i];
			const double& x = v[op.in[0]];
			return pow(a, x) * log(a);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{a^x, b^x}, {x}], {x}]
					{a^x log^2(a)}
					{b^x log^2(b)}	*/
			const double& a = op.constant[i];
			const double& x = v[op.in[0]];
			const double log_a = log(a);
			return pow(a, x) * log_a * log_a;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(constant.size()) };
	}

};

/****************************************************************************
*** POWER, SCALAR RAISED TO VECTOR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct PowerScalarVector;

template<> struct PowerScalarVector< true, true > : 
	OpIn::ArrayRangePair<1>, OpConst::None, OpOut::Range, 
	OpIsElementWise, OpOutSize::ArrayRange_None_RangeSize
{

	template< bool a, bool b> using family_t = PowerScalarVector< a, b >;

	PowerScalarVector(const OpIn::ArrayRangePair<1> in, const OpConst::None constant, const OpOut::Range out) :
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
			y = pow(a, b);
		}
	}

	struct LocalDiff {
		const PowerScalarVector< true, true >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < 1 + n);
			/* Grad[{a^x, a^y}, {a, x, y}]
				x a^(x-1)	a^x log(a)			0
				y a^(y-1)			0	a^y log(a) 	*/
			if (j > 0 && j != i + 1) return 0.0;
			const double& a = v[op.in.left[0]];
			const double& x = v[op.in.right[i]];
			if (j == 0) return x * pow(a, x - 1.);
			if (j == 1 + i) return pow(a, x) * log(a);
			return 0.0;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n + 1);
			assert(k < n + 1);
			/* Grad[Grad[{a^x, a^y}, {a, x, y}], {a, x, y}]

				{ (x-1) x a^(x-2), a^(x-1)+x a^(x-1) log(a),					   0}	{a^(x-1)+x a^(x-1) log(a), a^x log^2(a), 0}	{					    0, 0,			0}
				{ (y-1) y a^(y-2),						  0,a^(y-1)+y a^(y-1) log(a)}	{						0,			  0, 0}	{a^(y-1)+y a^(y-1) log(a), 0,a^y log^2(a)}			*/
			if (j == 0 && k == 0)
			{
				/* (x-1) x a^(x-2) */
				const double& a = v[op.in.left[0]];
				const double& x = v[op.in.right[i]];
				return (x - 1.) * x * pow(a, x - 2.);
			}
			if ((j == 0 && k == i + 1) || (j == i + 1 && k == 0))
			{
				/* a^(x-1)+x a^(x-1) log(a) */
				const double& a = v[op.in.left[0]];
				const double& x = v[op.in.right[i]];
				const double pow_a_xm1 = pow(a, x - 1.);
				return pow_a_xm1 + x * pow_a_xm1 * log(a);
			}
			if (j == i + 1 && k == i + 1) {
				/* a^x log^2(a) */
				const double& a = v[op.in.left[0]];
				const double& x = v[op.in.right[i]];
				const double log_a = log(a);
				return pow(a, x) * log_a * log_a;
			}
			else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(in.right.size()) };
	}

};


template<> struct PowerScalarVector< true, false > : 
	OpIn::Array<1>, OpConst::Vector, OpOut::Range,
	OpIsElementWise, OpOutSize::Scalar_Vector_VectorSize
{

	template< bool a, bool b> using family_t = PowerScalarVector< a, b >;

	PowerScalarVector(const OpIn::Array<1> in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Array<1>{ in }, OpConst::Vector{ constant }, OpOut::Range{ out }
	{
		assert(this->constant.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {

		const double& a = v[in[0]];

		for (index_t i = 0; i < index_t(constant.size()); i++) {
			double& y = v[out[i]];
			const double& b = constant[i];
			y = pow(a, b);
		}
	}
	struct LocalDiff {
		const PowerScalarVector< true, false >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j == 0);
			/* Grad[{a^x, a^y}, {a}]
				x a^(x-1)
				y a^(y-1) */
			const double& a = v[op.in[0]];
			const double& x = op.constant[i];
			return x * pow(a, x - 1.);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j == 0);
			assert(k == 0);
			/* Grad[Grad[{a^x, a^y}, {a}], {a}]

				{ (x-1) x a^(x-2) }
				{ (y-1) y a^(y-2) }			*/

			const double& a = v[op.in[0]];
			const double& x = op.constant[i];
			return (x - 1.) * x * pow(a, x - 2.);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(constant.size()) };
	}
};

template<> struct PowerScalarVector< false, true > : 
	OpIn::Range, OpConst::Array<1>, OpOut::Range, 
	OpIsElementWise, OpOutSize::Range_Scalar_RangeSize
{

	template< bool a, bool b> using family_t = PowerScalarVector< a, b >;

	PowerScalarVector(const OpIn::Range in, const OpConst::Array<1> constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Array<1>{ constant }, OpOut::Range{ out }
	{
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == out.size());
		const double& a = constant[0];
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& b = v[in[i]];
			y = pow(a, b);
		}
	}
	struct LocalDiff {
		const PowerScalarVector< false, true >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{a^x, a^y}, {x, y}]
				a^x log(a)			0
						0	a^y log(a) 	*/
			if (j != i) return 0.0;
			const double& a = op.constant[0];
			const double& x = v[op.in[i]];
			return pow(a, x) * log(a);
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/* Grad[Grad[{a^x, a^y}, {x, y}], {x, y}]

			{ a^x log^2(a), 0}	{ 0,			0}
			{		     0, 0}	{ 0, a^y log^2(a)}		*/
			if (j == i && k == i) {
				const double& a = op.constant[0];
				const double& x = v[op.in[i]];
				const double log_a = log(a);
				return pow(a, x) * log_a * log_a;
			}
			else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(in.size()) };
	}
};

/****************************************************************************
*** POWER, VECTOR RAISED TO VECTOR
*****************************************************************************/

/* WORK IN PROGRESS */
template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct PowerVectorVector;

template <> struct PowerVectorVector< true, true > : 
	OpIn::RangePair, OpConst::None, OpOut::Range, 
	OpIsElementWise, OpOutSize::RangePair_None_RangeSize
{

	template< bool a, bool b> using family_t = PowerVectorVector< a, b >;

	PowerVectorVector(const OpIn::RangePair in, const OpConst::None constant, const OpOut::Range out) : OpIn::RangePair{ in }, OpOut::Range{ out } {
		assert(this->in.left.size() == this->in.right.size());
		assert(this->out.size() == this->in.left.size());
		assert((this->in.right.begin() >= this->in.left.end()) || (this->in.left.begin() >= this->in.right.end()));
	}
	void evaluate(std::vector< double >& v) const {
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in.left[i]];
			const double& b = v[in.right[i]];
			y = pow(a, b);
		}
	}
	struct LocalDiff {
		const PowerVectorVector< true, true >& op;
		const std::vector< double >& v;
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < 2 * n);

			/* Grad[{a^x, b^y}, {a, b, x, y}]
			x a^(x-1)			0	a^x log(a)			0
					0	y b^(y-1)			 0	b^y log(b)		*/
			if ((j != i) && (j != i + n)) return 0.0;
			const double& a = v[op.in.left[i]];
			const double& x = v[op.in.right[i]];
			if (j == i) {
				/* x a^(x-1) */
				return x * pow(a, x - 1.);
			}
			else {
				/* a^x log(a) */
				return pow(a, x) * log(a);
			}
		}

		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < 2 * n);
			assert(k < 2 * n);
			/*
			Grad[Grad[{a^x, b^y}, {a, b, x, y}], {a, b, x, y}]

			j == 0
				{(x-1) x a^(x-2),0,a^(x-1)+x a^(x-1) log(a),0}			
				{0,0,0,0}			

			j == 1
			{0,0,0,0}
			{0,(y-1) y b^(y-2),0,b^(y-1)+y b^(y-1) log(b)}

			j == 2
			{a^(x-1)+x a^(x-1) log(a),0,a^x log^2(a),0}
			{0,0,0,0}

			j == 3
			{0,0,0,0}
			{0,b^(y-1)+y b^(y-1) log(b),0,b^y log^2(b)}

			*/
			if ((j != i) && (j != i + n)) return 0.0;
			if ((k != i) && (k != i + n)) return 0.0;
			const double& a = v[op.in.left[i]];
			const double& x = v[op.in.right[i]];
			if ((j == i) && (k == i)) {
				/* (x-1) x a^(x-2) */
				return (x - 1.) * x * pow(a, x - 2.);
			}
			if ( ((j == i) && (k == n + i)) ||
				 ((j == n + i) && (k == i)) )			
			{
				/* a^(x-1)+x a^(x-1) log(a) */
				const double pow_a_xm1 = pow(a, x - 1.);
				return pow_a_xm1 + x * pow_a_xm1 * log(a);
			}
			if ((j == n + i) && (k == n + i)) {
				/* a^x log^2(a) */
				const double log_a = log(a);
				return pow(a, x) * log_a * log_a;
			}
			else {
				assert(false);
				return NAN;
			}
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return { *this, v,  index_t(in.left.size()) };
	}
};

template<> struct PowerVectorVector< true, false > : 
	OpIn::Range, OpConst::Vector, OpOut::Range, 
	OpIsElementWise, OpOutSize::Range_Vector_RangeSize
{

	template< bool a, bool b> using family_t = PowerVectorVector< a, b >;

	PowerVectorVector(const OpIn::Range in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {
		assert(this->in.size() == this->constant.size());
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in[i]];
			const double& b = constant[i];
			y = pow(a, b);
		}
	}
	struct LocalDiff {
		const PowerVectorVector< true, false >& op;
		const std::vector< double >& v;
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);

			/* Grad[{a^x, b^y}, {a, b}]
			x a^(x-1)			0	
					0	y b^(y-1)	*/
			if (j != i) return 0.0;
			const double& a = v[op.in[i]];
			const double& x = op.constant[i];
			return x * pow(a, x - 1.);			
		}

		double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/*
			Grad[Grad[{a^x, b^y}, {a, b}], {a, b}]

				{ (x-1) x a^(x-2), 0 }	{ 0,			   0 }
				{				0, 0 }	{ 0, (y-1) y b^(y-2) }			*/
			if ((j != i) || (k != i)) return 0.0;
			const double& a = v[op.in[i]];
			const double& x = op.constant[i];
			return (x - 1.) * x * pow(a, x - 2.);
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return { *this, v,  index_t( in.size() ) };
	}
};

template<> struct PowerVectorVector< false, true > : 
	OpIn::Range, OpConst::Vector, OpOut::Range, 
	OpIsElementWise, OpOutSize::Range_Vector_RangeSize
{

	template< bool a, bool b> using family_t = PowerVectorVector< a, b >;

	PowerVectorVector(const OpIn::Range in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {
		assert(this->in.size() == this->constant.size());
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = constant[i];
			const double& b = v[in[i]];
			y = pow(a, b);
		}
	}
	struct LocalDiff {
		const PowerVectorVector< false, true >& op;
		const std::vector< double >& v;
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);

			/* Grad[{a^x, b^y}, {x, y}]
			a^x log(a)			0
					 0	b^y log(b)		*/
			if (j != i) return 0.0;
			const double& a = op.constant[i];
			const double& x = v[op.in[i]];
			return pow(a, x) * log(a);
		}

		double partial(const index_t i, const index_t j, const index_t k) {
			assert(i < n);
			assert(j < n);
			assert(k < n);
			/*
			Grad[Grad[{a^x, b^y}, {x, y}], {x, y}]

			{ a^x log^2(a), 0}	{ 0,			0}
			{		     0, 0}	{ 0, b^y log^2(b)}		*/
			if (j == i && k == i) {
				const double& a = op.constant[i];
				const double& x = v[op.in[i]];
				const double log_a = log(a);
				return pow(a, x) * log_a * log_a;
			}
			else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {		
		return { *this, v,  index_t( in.size() ) };
	}
};

/****************************************************************************
*** OPERATOR DECLARATION
*****************************************************************************/

using op_power_types = std::tuple<
	PowerScalarScalar< true, true >, PowerScalarScalar< true, false >, PowerScalarScalar< false, true >,
	PowerVectorScalar< true, true >, PowerVectorScalar< true, false >, PowerVectorScalar< false, true >,
	PowerScalarVector< true, true >, PowerScalarVector< true, false >, PowerScalarVector< false, true >,
	PowerVectorVector< true, true >, PowerVectorVector< true, false >, PowerVectorVector< false, true >
>;