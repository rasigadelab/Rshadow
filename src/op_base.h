/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once

/* 
Operators, a.k.a. Single Assignment Codes (SACs), are the basic building blocks of
automatically differentiable computational graphs.

Operators apply operations on elements of a Trace defined as input/output indices, see
OpIn and OpOut structs. They can also hold constant values, not part of the Trace, see
OpConst struct. Inputs and outputs can be multi-valued and their
gradient and Hessian are propagated through the Trace. Constant inputs can also be multi-valued
but their gradient and Hessian are ignored.

A sequence of operators is recorded in a Tape object (see tape.h) to define
arbitrarily complex functions.

A sequence of operators frequently used together can be recoded as a new, composite
operator. This may increase performance because each operation has a runtime overhead
due to value access (see Trace object in tape.h). However, composite operators should
maintain a reasonable number of inputs because the size of each operator in memory
equals the size of the largest operator and because all combinations of free/fixed inputs
must be declared (see op_binary.h for examples).

TRAIT REQUIREMENTS.

An operator:

-publicly inherits an input signature class (eg, OpIn::Array<1>),
	an output signature class (eg, OpOut::Scalar), and a constant
	input signature class (eg, OpConst::None)

-is constructed using the input and output signature classes, and the
	optional constant input class

-exposes an evaluate() method that computes the local function
	value and stores the value in the Trace::values vector at the position(s)
	defined in the OpOut field

-exposes a local_diff() method returning a LocalDiff
	object that exposes partial(i) and partial(i,j) methods to obtain
	1st and 2nd order partials. The rationale for moving the partial() methods
	out of the operator is to precompute common quantities before calling the
	partial() method repeatedly. Storing the precomputed quantities with the operator
	itself would increase the size of the Tape object, that store operators as a vector
	of variants.

Once a new operator is defined, its type must be added to the operator_types tuple
defined in operators.h. 

I/O signature classes:

-variable I/O expose in_field_t or out_field_t types that are iterable collections of index_t indices
	(std::array	or std::vector or a pair with begin index and size)
-constants obey the same design but have 'double' underlying type

*/

#include "utilities.h"
#include "tensor.h"

#include <array>
#include <vector>
#include <tuple>
#include <variant> // std::monostate

/****************************************************************************
*** TYPEDEFS AND HELPERS
*****************************************************************************/
/* For consistency with other parts of code */
//using index_t = long long;

/* Operand is free (or variable) */
const bool IN_FREE = true;
/* Operand is fixed (or constant) */
const bool IN_FIXED = false;


/* Operator tags indicate when local Hessian diagonal/offdiag is either always 0 or 1. This
is used to skip code paths and increase performance. */
struct PartialAlwaysZero {};
struct PartialAlwaysOne {};

struct HessianDiagAlwaysZero {};
struct HessianOffDiagAlwaysZero {};
struct HessianOffDiagAlwaysOne {};
struct HessianAlwaysZero : HessianDiagAlwaysZero, HessianOffDiagAlwaysZero {};

/* Operator tag for commutable operands */
struct OpIsCommutable {};

/* Tag for element-wise operators. An element-wise operator has the
following properties:
  -the output has size equal to the larger size of either input or constant
  -output element i only depends on scalar operands and/or the ith element
   in vector operands (either variable or constant). In other words, output
   element i is independent of any vector operand element j!=i.
  
  The OpIsElementWise tag determines the static code path taken in the
  OpVariantReverse() visitor. If present, the code path has complexity O(n)
  because we skip (i!=j) elements. If absent, the default code path is taken,
  with complexity O(n²) because all (i,j) pairs are considered.
*/
struct OpIsElementWise {};

/* Represents a range of values within a Tape/Trace vector */
class IndexRange {
protected:
	index_t begin_ = -1;
	index_t end_ = -1;
public:	
	index_t begin() const { return begin_; }
	index_t end() const { return end_; }
	size_t size() const { return end_ - begin_; }
	index_t operator[](const index_t i) const { return begin_ + i; }
	IndexRange(const index_t begin, const index_t end) : begin_{ begin }, end_{ end } {}
	IndexRange(const size_t begin, const size_t end) : begin_{ index_t(begin) }, end_{ index_t(end) } {}
	IndexRange(const int begin, const int end) : begin_{ index_t(begin) }, end_{ index_t(end) } {}
	IndexRange() = default;
};

/* Maximum allowable size for a static array in operator signature */
static constexpr index_t MAX_ARITY = 8;

/****************************************************************************
*** OPERATOR SIGNATURES - OPERANDS AND CONSTANTS
*****************************************************************************/

/* Operator signatures make a distinction between free operands, that are part
of the computational graph and that will be differentiated,
and constants that will not be differentiated. Constant 'belong' to the
operator while free inputs 'belong' to the Tape. */

/* UPDATE 2024-01-11 Refactored OpIn options in 3 categories Array, Vector and Range */

/* Input from trace, as an index or a collection of indices */
struct OpIn {

	/* Template for binary operands */
	template< typename Tleft, typename Tright >
	struct operand_pair_t {
		using left_t = Tleft;
		using right_t = Tright;
		left_t  left;
		right_t right;
		/* Vectorized {left, right} iteration */
		size_t size() const {
			return left.size() + right.size();
		}
		/* Vectorized random access */
		index_t operator[](const index_t i) const {
			if (i < index_t(left.size())) {
				return left[0] + i;
			}
			else {
				return right[0] + i - index_t(left.size());
			}
		}
	};
	
	/* Input is an ordered array of trace indices. Keep ARITY to a reasonable
	size: operators are stored in std::variant form (a type-safe union) so the largest
	operator size becomes the variant size. Prefer Range or Vector for large
	inputs. */
	template< size_t ARITY >
	struct Array {
		static_assert(ARITY == 1); // 2024-08-27 Removing Array<2>
		//static_assert(ARITY >  0); // Don't declare an empty array
		//static_assert(ARITY <= MAX_ARITY); // Avoid too large arrays, use Vector or Range type instead
		using in_field_t = std::array< index_t, ARITY >;
		using OpInType = Array< ARITY >;
		in_field_t in;		
		Array(const in_field_t& in) : in{ in } {}
		Array(const Array<1>&, const Array<1>&); // Specialized below
		Array(index_t); // Specialized below
		Array(index_t, index_t); // Specialized below
	};

	using Scalar = Array<1>;

	/* DEPRECATE ARRAY<2> */
	struct ScalarScalar {
		using in_field_t = operand_pair_t< Scalar::in_field_t, Scalar::in_field_t >;
		using OpInType = ScalarScalar;
		in_field_t in;
		ScalarScalar(const in_field_t& in) : in{ in } {}
		ScalarScalar(const Scalar& left, const Scalar& right) : in{ left.in[0], right.in[0] } {}
	};

	/* Input is a series of contiguous indices */
	struct Range {
		using in_field_t = IndexRange;
		using OpInType = Range;
		in_field_t in;
		Range(const in_field_t& in) : in{ in } {}
		Range(index_t begin, index_t end) : in{ begin, end } {}
		index_t operator[](index_t i) const { return in[i];	}
	};

	/* Inputs are two series of contiguous indices */
	struct RangePair {
		using in_field_t = operand_pair_t< IndexRange, IndexRange >;
		using OpInType = RangePair;
		in_field_t in;
	};

	/* Input is a series of contiguous indices and a scalar or fixed-size array  */
	template< size_t ARITY >
	struct RangeArrayPair {
		using OpInType = RangeArrayPair;
		using in_field_t = operand_pair_t< IndexRange, std::array< index_t, ARITY > >;
		in_field_t in;
		RangeArrayPair(const in_field_t& in) : in{ in } {}
		RangeArrayPair(const Range&, const Array<1>&); // Specialized below		
	};

	/* Input is a scalar or fixed-size array and a series of contiguous indices */
	template< size_t ARITY >
	struct ArrayRangePair {
		using OpInType = ArrayRangePair;
		using in_field_t = operand_pair_t< std::array< index_t, ARITY >, IndexRange >;
		in_field_t in;
		ArrayRangePair(const in_field_t& in) : in{ in } {}
		ArrayRangePair(const Array<1>& , const Range&); // Specialized below
	};

	/* Input is a generic Tensor with NDIM dimensions */
	template< size_t NDIM >
	struct InTensor {
		struct in_field_t : IndexRange {
			std::array< size_t, NDIM > dim;
			in_field_t(IndexRange rangebase, std::array< size_t, NDIM > dim) : 
				IndexRange{ rangebase }, dim{ dim } {}
		};
		using OpInType = InTensor;
		in_field_t in;
		InTensor(const in_field_t& in) : in{ in } {}
		/* Shorthand constructor */
		InTensor(IndexRange rangebase, std::array< size_t, NDIM > dim) :
			in(rangebase, dim) {}
	};

	/* Input is a pair of Tensors */
	template< size_t NDIM_LEFT, size_t NDIM_RIGHT >
	struct InTensorPair {
		using OpInType = InTensorPair;
		using left_t = typename OpIn::InTensor< NDIM_LEFT >::in_field_t;
		using right_t = typename OpIn::InTensor< NDIM_RIGHT >::in_field_t;
		using in_field_t = operand_pair_t< left_t, right_t >;
		in_field_t in;
		InTensorPair(const in_field_t& in) : in{ in } {}
		InTensorPair(OpIn::InTensor< NDIM_LEFT > left_in, OpIn::InTensor< NDIM_RIGHT > right_in) :
			in{ left_in.in, right_in.in } {}
	};
};

/*** SPECIALIZATIONS ***/
template<> inline OpIn::Array<1>::Array(index_t j) : in{ 1 } {
	in[0] = j;
}
template<> inline OpIn::RangeArrayPair<1>::RangeArrayPair(
	const OpIn::Range& left, const typename OpIn::Array<1>& right) :
	in{ left.in, right.in } {};
template<> inline OpIn::ArrayRangePair<1>::ArrayRangePair(
	const typename OpIn::Array<1>& left, const OpIn::Range& right) :
	in{ left.in, right.in } {};

/* Constant input (not part of trace). When inherited by an operator, exposes the
	.constant field*/
struct OpConst {

	/* There is no constant */
	struct None {
		using constant_field_t = std::monostate;
		using OpConstantType = None;
		constant_field_t constant;
	};

	/* A static array of constants */
	template< size_t ARITY >
	struct Array {
		static_assert(ARITY >  0); // Don't declare an empty array, use None instead
		static_assert(ARITY <= MAX_ARITY); // Avoid too large arrays, use Vector type instead
		using constant_field_t = std::array< double, ARITY >;
		using OpConstantType = Array< ARITY >;
		constant_field_t constant;
		Array(const Tensor& tensor) : constant{ { tensor.val[0] } } {
			assert(tensor.is_scalar());
		};
		Array(const Array< ARITY >& array) : constant{ array.constant } {}
		Array(const Array< ARITY >&& array) noexcept : constant{ array.constant } {}
		Array(const constant_field_t& constant) : constant{ constant } {}
		index_t operator[](index_t i) const { return constant[i]; }
	};



	/* A dynamic vector of constants */
	struct Vector {
		using constant_field_t = std::vector< double >;
		using OpConstantType = Vector;
		constant_field_t constant;
		Vector(const constant_field_t& constant) : constant{ constant } {}
		Vector(const Tensor& tensor) : constant{ tensor.val } {
			assert(tensor.is_vector());
		};
	};

	/* Constant is a generic Tensor with NDIM dimensions */
	template< size_t NDIM >
	struct ConstTensor {
		using constant_field_t = Tensor;
		using OpConstantType = ConstTensor;
		constant_field_t constant;
		ConstTensor(const Tensor& tensor) : constant{ tensor } {
			assert(tensor.dim.size() == NDIM);
		}
	};
};

/* Output to trace, as an index or a collection of indices */
struct OpOut {

	template< size_t ARITY >
	struct Array {
		static_assert(ARITY > 0); // Don't declare an empty array
		static_assert(ARITY <= MAX_ARITY); // Avoid too large arrays, use Range type instead
		using out_field_t = std::array< index_t, ARITY >;
		using OpOutType = Array< ARITY >;
		out_field_t out;
		template< typename T1, typename T2 >
		static constexpr size_t output_range_size(const T1&, const T2&) {
			return 1;
		}
	};

	using Scalar = Array<1>;

	struct Range {
		Range(const index_t begin, const index_t end) : out{ begin, end } {}
		Range(const IndexRange out) : out{ out } {}
		using out_field_t = IndexRange;
		using OpOutType = Range;
		out_field_t out;
		index_t operator[](index_t i) const { return out[i]; }
	};

};

/* Helpers to anticipate output range size from inputs.
Naming scheme is OpIn_OpConst_TypeOfOutputSize.
Would need (much) refactoring, being consistent with scalar, vector, matrix, tensor etc. */
struct OpOutSize {
	struct RangeArray_None_RangeSize {
		static size_t output_range_size(const OpIn::RangeArrayPair<1>& operand, const OpConst::None&) {
			return operand.in.left.size();
		}
	};
	struct ArrayRange_None_RangeSize {
		static size_t output_range_size(const OpIn::ArrayRangePair<1>& operand, const OpConst::None&) {
			return operand.in.right.size();
		}
	};
	struct Range_Scalar_RangeSize {
		static size_t output_range_size(const OpIn::Range& operand, const OpConst::Array<1>&) {
			return operand.in.size();
		}
	};
	struct Range_Vector_RangeSize {
		static size_t output_range_size(const OpIn::Range& operand, const OpConst::Vector&) {
			return operand.in.size();
		}
	};
	struct Range_None_RangeSize {
		static size_t output_range_size(const OpIn::Range& operand, const OpConst::None&) {
			return operand.in.size();
		}
	};
	struct Scalar_Vector_VectorSize {
		static size_t output_range_size(const OpIn::Array<1>&, const OpConst::Vector& op_constant) {
			return op_constant.constant.size();
		}
	};
	struct RangePair_None_RangeSize {
		static size_t output_range_size(const OpIn::RangePair& operand, const OpConst::None&) {
			return operand.in.left.size();
		}
	};
	/* C = A.B, A and B are free */
	struct MatrixPair_None_ProductSize {
		static size_t output_range_size(const OpIn::InTensorPair<2,2>& operand, const OpConst::None&) {
			/* Output of matrix product A*B has size rows(A) * cols(B) */
			return operand.in.left.dim[0] * operand.in.right.dim[1];
		}
	};
	/* C = A.B, A is free, B is fixed */
	struct Matrix_Matrix_LeftProductSize {
		static size_t output_range_size(const OpIn::InTensor<2>& left_operand, const OpConst::ConstTensor<2>& right_constant) {
			/* Output of matrix product A*B has size rows(A) * cols(B) */
			return left_operand.in.dim[0] * right_constant.constant.dim[1];
		}
	};
	/* C = A.B, A is fixed, B is free */
	struct Matrix_Matrix_RightProductSize {
		static size_t output_range_size(const OpIn::InTensor<2>& right_operand, const OpConst::ConstTensor<2>& left_constant) {
			/* Output of matrix product A*B has size rows(A) * cols(B) */
			return left_constant.constant.dim[0] * right_operand.in.dim[1];
		}
	};
};






