/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "op_base.h"

/****************************************************************************
*** ADDITION, SCALAR PLUS SCALAR
*****************************************************************************/

/* y = a + b */
template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct PlusScalarScalar;

/* y = a + b; a is free, b is free */
template<> struct PlusScalarScalar<true, true> : OpIn::ScalarScalar, OpConst::None, OpOut::Scalar, HessianAlwaysZero, OpIsCommutable {

	template< bool a, bool b> using family_t = PlusScalarScalar< a, b >;

	PlusScalarScalar(const OpIn::ScalarScalar in, const OpConst::None constant, const OpOut::Scalar out) : OpIn::ScalarScalar{ in }, OpOut::Scalar{ out } {
		assert((this->in.left[0] != this->in.right[0]) && "Use MultiplyScalarScalar< true, false > 2 operator instead.");
	}
	void evaluate(std::vector< double >& v) const {
		v[out[0]] = v[in.left[0]] + v[in.right[0]];
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0 || j == 1);
			return 1.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0 || j == 1);
			assert(k == 0 || k == 1);
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{};
	}
};

/* y = a + b; a is free, b is fixed */
template<> struct PlusScalarScalar<true, false> : OpIn::Array<1>, OpConst::Array<1>, OpOut::Scalar, HessianAlwaysZero, OpIsCommutable {

	template< bool a, bool b> using family_t = PlusScalarScalar< a, b >;

	PlusScalarScalar(const OpIn::Array<1> in, const OpConst::Array<1> constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpOut::Scalar{ out }, OpConst::Array<1>{ constant } {}

	void evaluate(std::vector< double >& v) const {
		v[out[0]] = v[in[0]] + constant[0];
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
		return LocalDiff{};
	}
};

/* Equivalent, commuted unary operator. See
https://en.cppreference.com/w/cpp/language/using_declaration
for the templated constructor inheritance pattern */
template<> struct PlusScalarScalar<false, true> : PlusScalarScalar<true, false> {
	using PlusScalarScalar<true, false>::PlusScalarScalar;
};

/****************************************************************************
*** ADDITION, VECTOR PLUS SCALAR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct PlusVectorScalar;

template<> struct PlusVectorScalar< true, true > : 
	OpIn::RangeArrayPair<1>, OpConst::None, OpOut::Range, HessianAlwaysZero, 
	OpIsElementWise, OpOutSize::RangeArray_None_RangeSize
{

	template< bool a, bool b> using family_t = PlusVectorScalar< a, b >;

	PlusVectorScalar(const OpIn::RangeArrayPair<1> in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::RangeArrayPair<1>{ in }, OpOut::Range{ out } 
	{
		assert((this->in.left.end() <= this->in.right[0]) || (this->in.right[0] < this->in.left.begin()));
	}
	void evaluate(std::vector< double >& v) const {
		const double& b = v[in.right[0]];
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in.left[i]];
			y = a + b;
		}
	}
	struct LocalDiff {
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{a+x, b+x}, {a, b, x}] */
			/*
			1	0	1
			0	1	1
			*/

			/* given y=a+b, the partial is 1 iif i is the index of a or b

			After vectorization, this implies that j = i or j = n
			*/
			if (j == i || j == n) return 1.0;
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

template<> struct PlusVectorScalar< true, false > : 
	OpIn::Range, OpConst::Array<1>, OpOut::Range, 
	HessianAlwaysZero, OpIsElementWise, OpOutSize::Range_Scalar_RangeSize
{

	template< bool a, bool b> using family_t = PlusVectorScalar< a, b >;

	PlusVectorScalar(const OpIn::Range in, const OpConst::Array<1> constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Array<1>{ constant }, OpOut::Range{ out } {}

	void evaluate(std::vector< double >& v) const {
		const double& b = constant[0];
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in[i]];
			y = a + b;
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

template<> struct PlusVectorScalar< false, true > : 
	OpIn::Array<1>, OpConst::Vector, OpOut::Range, 
	HessianAlwaysZero, OpIsElementWise, OpOutSize::Scalar_Vector_VectorSize
{

	template< bool a, bool b> using family_t = PlusVectorScalar< a, b >;

	PlusVectorScalar(const OpIn::Array<1> in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Array<1>{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {}

	void evaluate(std::vector< double >& v) const {
		const double& b = v[in[0]];
		for (index_t i = 0; i < index_t(constant.size()); i++) {
			double& y = v[out[i]];
			y = constant[i] + b;
		}
	}
	struct LocalDiff {
		constexpr double partial(const index_t i, const index_t j) {
			assert(j == 0);
			return 1.0;
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
*** ADDITION, SCALAR PLUS VECTOR
*****************************************************************************/

/* These are not defined because of operator commutativity */

/****************************************************************************
*** ADDITION, VECTOR PLUS VECTOR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct PlusVectorVector;

template<> struct PlusVectorVector< true, true > : 
	OpIn::RangePair, OpConst::None, OpOut::Range, 
	HessianAlwaysZero, OpIsElementWise, OpOutSize::RangePair_None_RangeSize
{

	template< bool a, bool b> using family_t = PlusVectorVector< a, b >;

	PlusVectorVector(const OpIn::RangePair in, const OpConst::None constant, const OpOut::Range out) : OpIn::RangePair{ in }, OpOut::Range{ out } {
		assert(this->in.left.size() == this->in.right.size());
		assert(this->in.left.size() == this->out.size());
		assert((this->in.right.begin() >= this->in.left.end()) || (this->in.left.begin() >= this->in.right.end()));
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.left.size() == in.right.size());
		assert(in.left.size() == out.size());
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in.left[i]];
			const double& b = v[in.right[i]];
			y = a + b;
		}
	}
	struct LocalDiff {
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			/* Grad[{a+x, b+y, c+z}, {a, b, c, x, y, z}] */
			/*
			1	0	0	1	0	0
			0	1	0	0	1	0
			0	0	1	0	0	1
			*/

			/* given y=a+b, the partial is 1 iif i is the index of a or b

			After vectorization, this implies that j = i or j = i + n
			*/
			if (i == j || i + n == j) return 1.0; 
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

template<> struct PlusVectorVector< true, false > : 
	OpIn::Range, OpConst::Vector, OpOut::Range, 
	HessianAlwaysZero, OpIsElementWise, OpOutSize::Range_Vector_RangeSize
{

	template< bool a, bool b> using family_t = PlusVectorVector< a, b >;

	PlusVectorVector(const OpIn::Range in, const OpConst::Vector constant, const OpOut::Range out) :
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
			y = a + b;
		}
	}
	struct LocalDiff {
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			assert(i < index_t(n));
			assert(j < index_t(n));
			/* Grad[{a+x, b+y}, {a, b}]
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

template<> struct PlusVectorVector<false, true> : PlusVectorVector<true, false> {
	using PlusVectorVector<true, false>::PlusVectorVector;
};

/****************************************************************************
*** OPERATOR DECLARATION
*****************************************************************************/

using op_addition_types = std::tuple<
	PlusScalarScalar< true, true >, PlusScalarScalar< true, false >, PlusScalarScalar< false, true >,
	PlusVectorScalar< true, true >, PlusVectorScalar< true, false >, PlusVectorScalar< false, true >,
	PlusVectorVector< true, true >, PlusVectorVector< true, false >, PlusVectorVector< false, true >
>;