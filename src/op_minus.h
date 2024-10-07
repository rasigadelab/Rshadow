/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "op_base.h"

/****************************************************************************
*** SCALAR MINUS SCALAR
*****************************************************************************/

/* y = a - b */
template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct MinusScalarScalar;

/* y = a - b; a is free, b is free */
template<> struct MinusScalarScalar< true, true > :
	OpIn::ScalarScalar, OpConst::None, OpOut::Scalar, HessianAlwaysZero {

	template< bool a, bool b> using family_t = MinusScalarScalar< a, b >;

	MinusScalarScalar(const OpIn::ScalarScalar in, const OpConst::None constant, const OpOut::Scalar out) : OpIn::ScalarScalar{ in }, OpOut::Scalar{ out } {
		assert(this->in.left[0] > -1 && this->in.right[0] > -1);
		assert((this->in.left[0] != this->in.right[0]) && "Duplicate edge detected. Please use Trivial<0> operator instead.");
	}

	void evaluate(std::vector< double >& v) const {
		v[out[0]] = v[in.left[0]] - v[in.right[0]];
	}

	struct LocalDiff {
		double partial(const index_t i, const index_t j) const {
			assert(j == 0 || j == 1);
			return j == 1 ? -1. : 1.;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(j == 0 || j == 1);
			assert(k == 0 || k == 1);
			return 0.0;
		}
	};

	LocalDiff local_diff(const std::vector< double >& v) const {
		return {};
	}
};

/* y = a - b; a is free, b is fixed */
template<> struct MinusScalarScalar< true, false > :
	OpIn::Array<1>, OpConst::Array<1>, OpOut::Scalar, HessianAlwaysZero {

	template< bool a, bool b> using family_t = MinusScalarScalar< a, b >;

	MinusScalarScalar(const OpIn::Array<1> in, const OpConst::Array<1> constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out }, OpConst::Array<1>{ constant } {}

	void evaluate(std::vector< double >& v) const {
		v[out[0]] = v[in[0]] - constant[0];
	}

	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			return 1.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			return 0.0;
		}
	};

	LocalDiff local_diff(const std::vector< double >& v) const {
		return {};
	}
};

/* y = a - b; a is fixed, b is free */
template<> struct MinusScalarScalar< false, true > :
	OpIn::Array<1>, OpConst::Array<1>, OpOut::Scalar, HessianAlwaysZero {

	template< bool a, bool b> using family_t = MinusScalarScalar< a, b >;

	MinusScalarScalar(const OpIn::Array<1> in, const OpConst::Array<1> constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out }, OpConst::Array<1>{ constant } {}

	void evaluate(std::vector< double >& v) const {
		v[out[0]] = constant[0] - v[in[0]];
	}

	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			return -1.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			return 0.0;
		}
	};

	LocalDiff local_diff(const std::vector< double >& v) const {
		return {};
	}
};

/****************************************************************************
*** VECTOR MINUS SCALAR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct MinusVectorScalar;

template<> struct MinusVectorScalar< true, true > : 
	OpIn::RangeArrayPair<1>, OpConst::None, OpOut::Range, HessianAlwaysZero,
	OpIsElementWise, OpOutSize::RangeArray_None_RangeSize
{

	template< bool a, bool b> using family_t = MinusVectorScalar< a, b >;

	MinusVectorScalar(const OpIn::RangeArrayPair<1> in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::RangeArrayPair<1>{ in }, OpOut::Range{ out }
	{
		assert((this->in.left.end() <= this->in.right[0]) || (this->in.right[0] < this->in.left.begin()));
	}


	void evaluate(std::vector< double >& v) const {
		const double& b = v[in.right[0]];
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in.left[i]];
			y = a - b;
		}
	}
	struct LocalDiff {
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {

			/* Grad[{a-x, b-x}, {a, b, x}] */
			/*
			1	0	-1
			0	1	-1
			*/
			if (j == i) return  1.0;
			if (j == n) return -1.0;
			else return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) {
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ index_t(in.left.size()) };
	}
};

template<> struct MinusVectorScalar< true, false > :
	OpIn::Range, OpConst::Array<1>, OpOut::Range, HessianAlwaysZero,
	OpIsElementWise, OpOutSize::Range_Scalar_RangeSize
{

	template< bool a, bool b> using family_t = MinusVectorScalar< a, b >;

	MinusVectorScalar(const OpIn::Range in, const OpConst::Array<1> constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Array<1>{ constant }, OpOut::Range{ out } {}

	void evaluate(std::vector< double >& v) const {
		const double& b = constant[0];
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in[i]];
			y = a - b;
		}
	}
	struct LocalDiff {
		double partial(const index_t i, const index_t j) const {
			if (j == i) return 1.0;
			else return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};

template<> struct MinusVectorScalar< false, true > : 
	OpIn::Array<1>, OpConst::Vector, OpOut::Range, HessianAlwaysZero, 
	OpIsElementWise, OpOutSize::Scalar_Vector_VectorSize
{

	template< bool a, bool b> using family_t = MinusVectorScalar< a, b >;

	MinusVectorScalar(const OpIn::Array<1> in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Array<1>{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {}

	void evaluate(std::vector< double >& v) const {
		const double& b = v[in[0]];
		for (index_t i = 0; i < index_t(constant.size()); i++) {
			double& y = v[out[i]];
			y = constant[i] - b;
		}
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) {
			assert(j == 0);
			return -1.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) {
			assert(j == 0);
			assert(k == 0);
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};

/****************************************************************************
*** SCALAR MINUS VECTOR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct MinusScalarVector;

template<> struct MinusScalarVector< true, true > : 
	OpIn::ArrayRangePair<1>, OpConst::None, OpOut::Range, HessianAlwaysZero, 
	OpIsElementWise, OpOutSize::ArrayRange_None_RangeSize
{

	template< bool a, bool b> using family_t = MinusScalarVector< a, b >;

	MinusScalarVector(const OpIn::ArrayRangePair<1> in, const OpConst::None constant, const OpOut::Range out) :
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
			y = a - b;
		}
	}
	struct LocalDiff {
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < 1 + n);
			/* Grad[{x-a, x-b}, {x, a, b}]
			1	-1	 0
			1	 0	-1
			*/
			if (j == i + 1) return -1.0;
			if (j == 0) return  1.0;
			return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) {
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ index_t(in.right.size()) };
	}
};

template<> struct MinusScalarVector< true, false > : 
	OpIn::Array<1>, OpConst::Vector, OpOut::Range, HessianAlwaysZero, 
	OpIsElementWise, OpOutSize::Scalar_Vector_VectorSize
{

	template< bool a, bool b> using family_t = MinusScalarVector< a, b >;

	MinusScalarVector(const OpIn::Array<1> in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Array<1>{ in }, OpConst::Vector{ constant }, OpOut::Range{ out }
	{
		assert(this->constant.size() == this->out.size());
	}

	void evaluate(std::vector< double >& v) const {

		const double& a = v[in[0]];

		for (index_t i = 0; i < index_t(constant.size()); i++) {
			double& y = v[out[i]];
			const double& b = constant[i];
			y = a - b;
		}
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			return 1.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) {
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {		
		return LocalDiff{};
	}
};

template<> struct MinusScalarVector< false, true > : 
	OpIn::Range, OpConst::Array<1>, OpOut::Range, HessianAlwaysZero, 
	OpIsElementWise, OpOutSize::Range_Scalar_RangeSize
{

	template< bool a, bool b> using family_t = MinusScalarVector< a, b >;

	MinusScalarVector(const OpIn::Range in, const OpConst::Array<1> constant, const OpOut::Range out) :
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
			y = a - b;
		}
	}
	struct LocalDiff {
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{C-x, C-y}, {x, y}]
			-1	 0
			 0	-1
			*/
			if (i == j) return -1.0;
			else return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) {
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ index_t(in.size()) };
	}
};

/****************************************************************************
*** VECTOR MINUS VECTOR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct MinusVectorVector;

template<> struct MinusVectorVector< true, true > : 
	OpIn::RangePair, OpConst::None, OpOut::Range, 
	HessianAlwaysZero, OpIsElementWise, OpOutSize::RangePair_None_RangeSize
{

	template< bool a, bool b> using family_t = MinusVectorVector< a, b >;

	MinusVectorVector(const OpIn::RangePair in, const OpConst::None constant, const OpOut::Range out) : OpIn::RangePair{ in }, OpOut::Range{ out } {
		assert(this->in.left.size() == this->in.right.size());
		assert((this->in.right.begin() >= this->in.left.end()) || (this->in.left.begin() >= this->in.right.end()));
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.left.size() == in.right.size());
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in.left[i]];
			const double& b = v[in.right[i]];
			y = a - b;
		}
	}
	struct LocalDiff {
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < 2 * n);
			/* Grad[{a-x, b-y, c-z}, {a, b, c, x, y, z}] */
			/*
			1	0	0	-1	 0	 0
			0	1	0	 0	-1	 0
			0	0	1	 0	 0	-1
			*/
			if (j == i) return 1.0;
			if (j == i + n) return -1.0;
			else return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) {
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		assert(in.left.size() == in.right.size());
		return LocalDiff{ index_t(in.left.size()) };
	}
};

template<> struct MinusVectorVector< true, false > : 
	OpIn::Range, OpConst::Vector, OpOut::Range, 
	HessianAlwaysZero, OpIsElementWise, OpOutSize::Range_Vector_RangeSize
{

	template< bool a, bool b> using family_t = MinusVectorVector< a, b >;

	MinusVectorVector(const OpIn::Range in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {
		assert(this->in.size() == this->constant.size());
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == constant.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in[i]];
			const double& b = constant[i];
			y = a - b;
		}
	}
	struct LocalDiff {
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{a-x, b-y}, {a, b}]
			1	0
			0	1
			*/
			if (i == j) return 1.0;
			else return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) {
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		assert(in.size() == constant.size());
		return LocalDiff{ index_t(in.size()) };
	}
};

template<> struct MinusVectorVector< false, true > : 
	OpIn::Range, OpConst::Vector, OpOut::Range, 
	HessianAlwaysZero, OpIsElementWise, OpOutSize::Range_Vector_RangeSize
{

	template< bool a, bool b> using family_t = MinusVectorVector< a, b >;

	MinusVectorVector(const OpIn::Range in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {
		assert(this->in.size() == this->constant.size());
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.size() == constant.size());
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = constant[i];
			const double& b = v[in[i]];
			y = a - b;
		}
	}
	struct LocalDiff {
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			assert(i < n);
			assert(j < n);
			/* Grad[{a-x, b-y}, {x, y}]
			-1	 0
			 0	-1
			*/
			if (i == j) return -1.0;
			else return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) {
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		assert(in.size() == constant.size());
		return LocalDiff{ index_t(in.size()) };
	}
};

/****************************************************************************
*** OPERATOR DECLARATION
*****************************************************************************/

using op_minus_types = std::tuple<
	MinusScalarScalar< true, true >, MinusScalarScalar< true, false >, MinusScalarScalar< false, true >,
	MinusVectorScalar< true, true >, MinusVectorScalar< true, false >, MinusVectorScalar< false, true >,
	MinusScalarVector< true, true >, MinusScalarVector< true, false >, MinusScalarVector< false, true >,
	MinusVectorVector< true, true >, MinusVectorVector< true, false >, MinusVectorVector< false, true >
>;