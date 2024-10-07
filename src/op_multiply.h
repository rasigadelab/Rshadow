/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "op_base.h"

/****************************************************************************
*** MULTIPLICATION, SCALAR TIMES SCALAR
*****************************************************************************/

/* y = a * b */
template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct MultiplyScalarScalar;

/* y = a * b; a is free, b is free */
template<> struct MultiplyScalarScalar<true, true> :
	OpIn::ScalarScalar, OpConst::None, OpOut::Scalar, HessianDiagAlwaysZero, HessianOffDiagAlwaysOne, OpIsCommutable {

	template< bool a, bool b> using family_t = MultiplyScalarScalar< a, b >;

	MultiplyScalarScalar(const OpIn::ScalarScalar in, const OpConst::None constant, const OpOut::Scalar out) : OpIn::ScalarScalar{ in }, OpOut::Scalar{ out } {
		assert((this->in.left[0] != this->in.right[0]) && "Duplicate edge detected. Please use Square operator instead.");
	}
	void evaluate(std::vector< double >& v) const {
		v[out[0]] = v[in.left[0]] * v[in.right[0]];
	}
	struct LocalDiff {
		std::array< double, 2 > x = { 0.0, 0.0 };
		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0 || j == 1);
			return x[j];
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[ a * b, {a, b} ], {a, b}]
			0 1
			1 0
			*/
			assert(i == 0);
			assert(j == 0 || j == 1);
			assert(k == 0 || k == 1);
			return (j == k) ? 0.0 : 1.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		/* Remark that the values are swapped here but not in the Diff object */
		return LocalDiff{ v[in.right[0]], v[in.left[0]] };
	}
};


/* y = a * b; a is free, b is fixed */
template<> struct MultiplyScalarScalar<true, false> : OpIn::Array<1>, OpConst::Array<1>, OpOut::Scalar,
	HessianDiagAlwaysZero, OpIsCommutable
{

	template< bool a, bool b> using family_t = MultiplyScalarScalar< a, b >;

	MultiplyScalarScalar(const OpIn::Array<1> in, const OpConst::Array<1> constant, const OpOut::Scalar out) :
		OpIn::Array<1>{ in }, OpConst::Array<1>{ constant }, OpOut::Scalar{ out } {}

	void evaluate(std::vector< double >& v) const {
		v[out[0]] = v[in[0]] * constant[0];
	}
	struct LocalDiff {
		const double c = 0.0; // The operator's constant factor

		double partial(const index_t i, const index_t j) const {
			assert(i == 0);
			assert(j == 0);
			return c;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			assert(i == 0);
			assert(j == 0);
			assert(k == 0);
			return 0.0;
		}
	};

	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ constant[0] };
	}

};

/* y = a * b; a is fixed, b is free */
template<> struct MultiplyScalarScalar<false, true> : MultiplyScalarScalar<true, false> {
	using MultiplyScalarScalar<true, false>::MultiplyScalarScalar;
};

/****************************************************************************
*** MULTIPLICATION, VECTOR TIMES SCALAR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct MultiplyVectorScalar;

template<> struct MultiplyVectorScalar< true, true > : 
	OpIn::RangeArrayPair<1>, OpConst::None, OpOut::Range, 
	OpIsElementWise, OpOutSize::RangeArray_None_RangeSize
{

	template< bool a, bool b> using family_t = MultiplyVectorScalar< a, b >;

	MultiplyVectorScalar(const OpIn::RangeArrayPair<1> in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::RangeArrayPair<1>{ in }, OpOut::Range{ out }
	{
		assert((this->in.left.end() <= this->in.right[0]) || (this->in.right[0] < this->in.left.begin()));
	}
	void evaluate(std::vector< double >& v) const {
		const double& b = v[in.right[0]];
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in.left[i]];
			y = a * b;
		}
	}
	struct LocalDiff {
		const MultiplyVectorScalar< true, true >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{a*x, b*x}, {a, b, x}]
			x	0	a
			0	x	b
			*/
			if (j == i) return v[op.in.right[0]]; // x
			if (j == n) return v[op.in.left[i]];  // a or b
			else return 0.0;
		}
		double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[{a*x, b*x}, {a, b, x}], {a, b, x}]

			  {0, 0, 1}, {0, 0, 0}, {1, 0, 0}
			  {0, 0, 0}, {0, 0, 1}, {0, 1, 0}

			  ROW is i, index in output vector
			  COLUMN is j, index in (vectorized) left input
			  POSITION IN VECTOR is k, index in (vectorized) right input
			*/
			if (j == i && k == n) return 1.0;
			if (j == n && k == i) return 1.0;
			else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(in.left.size()) };
	}
};

template<> struct MultiplyVectorScalar< true, false > : 
	OpIn::Range, OpConst::Array<1>, OpOut::Range, 
	OpIsElementWise, OpOutSize::Range_Scalar_RangeSize
{

	template< bool a, bool b> using family_t = MultiplyVectorScalar< a, b >;

	MultiplyVectorScalar(const OpIn::Range in, const OpConst::Array<1> constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Array<1>{ constant }, OpOut::Range{ out } {}

	void evaluate(std::vector< double >& v) const {
		const double& b = constant[0];
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in[i]];
			y = a * b;
		}
	}
	struct LocalDiff {

		const double c = 0.0; // The scalar constant
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {
			/* Grad[{a*c, b*c}, {a, b}]
			c	0
			0	c
			*/
			assert(i < n);
			assert(j < n);
			if (j == i) return c;
			else return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[{a*c, b*c}, {a, b}], {a, b}] */
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ constant[0], index_t(in.size()) };
	}
};

template<> struct MultiplyVectorScalar< false, true > : 
	OpIn::Array<1>, OpConst::Vector, OpOut::Range, 
	OpIsElementWise, OpOutSize::Scalar_Vector_VectorSize 
{

	template< bool a, bool b> using family_t = MultiplyVectorScalar< a, b >;

	MultiplyVectorScalar(const OpIn::Array<1> in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Array<1>{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {}

	void evaluate(std::vector< double >& v) const {
		const double& b = v[in[0]];
		for (index_t i = 0; i < index_t(constant.size()); i++) {
			double& y = v[out[i]];
			y = constant[i] * b;
		}
	}
	struct LocalDiff {
		const MultiplyVectorScalar< false, true >& op;
		const std::vector< double >& v;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{a*x, b*x}, {x}]
			a
			b
			*/
			assert(i < n);
			assert(j == 0);
			return op.constant[i];
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[{a*x, b*x}, {x}], {x}] */
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v, index_t(constant.size()) };
	}
};

/****************************************************************************
*** MULTIPLICATION, SCALAR TIMES VECTOR
*****************************************************************************/

/* These are not defined because of operator commutativity */

/****************************************************************************
*** MULTIPLICATION, VECTOR TIMES VECTOR
*****************************************************************************/

template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct MultiplyVectorVector;

template<> struct MultiplyVectorVector< true, true > : 
	OpIn::RangePair, OpConst::None, OpOut::Range, 
	OpIsElementWise, OpOutSize::RangePair_None_RangeSize
{

	template< bool a, bool b> using family_t = MultiplyVectorVector< a, b >;

	MultiplyVectorVector(const OpIn::RangePair in, const OpConst::None constant, const OpOut::Range out) : OpIn::RangePair{ in }, OpOut::Range{ out } {
		assert(this->in.left.size() == this->in.right.size());
		assert((this->in.right.begin() >= this->in.left.end()) || (this->in.left.begin() >= this->in.right.end()));
	}
	void evaluate(std::vector< double >& v) const {
		assert(in.left.size() == in.right.size());
		for (index_t i = 0; i < index_t(in.left.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in.left[i]];
			const double& b = v[in.right[i]];
			y = a * b;
		}
	}
	struct LocalDiff {
		const MultiplyVectorVector< true, true >& op;
		const std::vector< double >& v;
		const index_t n = 0;

		double partial(const index_t i, const index_t j) const {

			/* Grad[{a*x, b*y, c*z}, {a, b, c, x, y, z}] 
			x	0	0	a	0	0
			0	y	0	0	b	0
			0	0	z	0	0	c
			*/
			/* Partial wrt left element is right element */
			if (j == i) return v[op.in.right[j]];
			/* Partial wrt right element is left element */
			if (j == i + n) return v[op.in.left[i]];
			else return 0.0;
		}

		double partial(const index_t i, const index_t j, const index_t k) const {
			/*
			Grad[Grad[{a*x, b*y, c*z}, {a, b, c, x, y, z}], {a, b, c, x, y, z}]

			{0,0,0,1,0,0}	{0,0,0,0,0,0}	{0,0,0,0,0,0}	{1,0,0,0,0,0}	{0,0,0,0,0,0}	{0,0,0,0,0,0}
			{0,0,0,0,0,0}	{0,0,0,0,1,0}	{0,0,0,0,0,0}	{0,0,0,0,0,0}	{0,1,0,0,0,0}	{0,0,0,0,0,0}
			{0,0,0,0,0,0}	{0,0,0,0,0,0}	{0,0,0,0,0,1}	{0,0,0,0,0,0}	{0,0,0,0,0,0}	{0,0,1,0,0,0}

			Unit partial iif i == w AND j == w + n or the other way around (commutability)

						ROW is i
						OUTER COLUMN is j
						INNER COLUMN is k (position in vector, equivalent to 3rd array dimension)
			*/

			if ((j == i) && (k == i + n)) return 1.0;
			if ((j == i + n) && (k == i)) return 1.0;
			else return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		assert(in.left.size() == in.right.size());
		return LocalDiff{ *this, v, index_t(in.left.size()) };
	}
};

template<> struct MultiplyVectorVector< true, false > : 
	OpIn::Range, OpConst::Vector, OpOut::Range, 
	OpIsElementWise, OpOutSize::Range_Vector_RangeSize
{

	template< bool a, bool b> using family_t = MultiplyVectorVector< a, b >;

	MultiplyVectorVector(const OpIn::Range in, const OpConst::Vector constant, const OpOut::Range out) :
		OpIn::Range{ in }, OpConst::Vector{ constant }, OpOut::Range{ out } {
		assert(this->in.size() == this->constant.size());
		assert(this->in.size() == this->out.size());
	}
	void evaluate(std::vector< double >& v) const {
		for (index_t i = 0; i < index_t(in.size()); i++) {
			double& y = v[out[i]];
			const double& a = v[in[i]];
			const double& b = constant[i];
			y = a * b;
		}
	}
	struct LocalDiff {
		const MultiplyVectorVector< true, false >& op;
		const index_t n = 0;
		double partial(const index_t i, const index_t j) const {
			/* Grad[{a*x, b*y}, {a, b}]
			x  0
			0  y
			*/
			assert(i < n);
			assert(j < n);
			if (j == i) return op.constant[i];
			else return 0.0;
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			/* Grad[Grad[{a*x, b*y}, {a, b}], {a, b}] */
			return 0.0;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, index_t(constant.size()) };
	}
};

template<> struct MultiplyVectorVector<false, true> : MultiplyVectorVector<true, false> {
	using MultiplyVectorVector<true, false>::MultiplyVectorVector;
};

/****************************************************************************
*** MATRIX PRODUCT
*****************************************************************************/

/* y = a * b */
template< bool A_IS_FREE = true, bool B_IS_FREE = true > struct MultiplyMatrixMatrix;

/* y = a * b; a is free, b is free */
template<> struct MultiplyMatrixMatrix<true, true> :
	OpIn::InTensorPair<2, 2>, OpConst::None, OpOut::Range,
	OpOutSize::MatrixPair_None_ProductSize
{
	template< bool a, bool b> using family_t = MultiplyMatrixMatrix< a, b >;

	MultiplyMatrixMatrix(const OpIn::InTensorPair<2, 2> in, const OpConst::None constant, const OpOut::Range out) :
		OpIn::InTensorPair<2, 2>{ in }, OpOut::Range{ out } {
		assert((this->in.left[0] != this->in.right[0]) && "Duplicate edge detected. Please use Square operator instead.");
	}
	void evaluate(std::vector< double >& v) const {

		const size_t a_nrows = in.left.dim[0];
		const size_t b_ncols = in.right.dim[1];
		/* cols(A) = rows(B) */
		const size_t K = in.right.dim[0];
		assert(K == in.left.dim[0]);

		/* ROW LOOP */
		for (size_t row = 0; row < a_nrows; row++) {
			/* COLUMN LOOP */
			for (size_t col = 0; col < b_ncols; col++) {
				double x = 0.;
				/* INNER LOOP */
				for (size_t k = 0; k < K; k++) {
					const double& a_ijk = v[in.left[row + k * a_nrows]];
					const double& b_ijk = v[in.right[k + col * K]];
					x += a_ijk * b_ijk;
				}
				v[out[0] + row + col * a_nrows] = x;
			}
		}
	}
	struct LocalDiff {
		const MultiplyMatrixMatrix< true, true >& op;
		const std::vector< double >& v;

		double partial(const index_t i, const index_t j) const {
			/* USING COLUMN ORDER VEC OPERATOR
			* |a c|.|e g|=| ae+cf ag+ch|     <=>  A.B = C
			* |b d| |f h| | be+df bg+dh|
			*
			Grad[ { a e + c f, b e + d f, a g + c h, b g + d h }, {a,b,c,d,e,f,g,h} ]

			Grad[{{ a, c}, {b, d}}.{{ e, g}, {f, h}}, {a,b,c,d,e,f,g,h}] =

			|{e,0,f,0,a,c,0,0}	{g,0,h,0,0,0,a,c}|
			|{0,e,0,f,b,d,0,0}	{0,g,0,h,0,0,b,d}|

			Vectorization:
			i = row + col * rows(C)
			j = row + col * rows(A)^[in A?] * rows(B)^[in B?] + [in B?] * size(A)

			General Jacobian formula:
			dC_kl / dA_ij = B_jl if i==k, 0 otherwise
			dC_kl / dB_ij = A_ki if j==l, 0 otherwise

			For readability we use index names prefixed with matrix name,
			A(ai, aj) B(bi, bj) C(ci cj)


			A readable approach would be to define "devectorized" indices i_, j_, k_, l_
			from vectorized input indices i,j, as well the operand index (0 for A, 1 for B)
			*/

			const index_t rows_A = op.in.left.dim[0];
			const index_t cols_A = op.in.left.dim[1];
			const index_t rows_B = op.in.right.dim[0];
			const index_t rows_C = rows_A;
			const index_t size_A = rows_A * cols_A;

			const bool j_is_in_B = (j >= size_A);
			const index_t rows_operand = j_is_in_B ? rows_B : rows_A;
			/* Map vectorized operand element index j onto matrix index X(xi, xj) (X may be A or B)*/
			const index_t j_aligned = j_is_in_B ? j - size_A : j;
			const index_t xi = j_aligned % rows_operand;
			const index_t xj = j_aligned / rows_operand;
			/* Map vectorized result element index i onto matrix index C(ci, cj) */
			const index_t ci = i % rows_C;
			const index_t cj = i / rows_C;

			if (j_is_in_B == 0) { // element is in A
				if (xi != ci) return 0.0;
				/* Vectorized index in B: (aj, cj) => vb */
				const index_t vb = xj + cj * rows_B;
				return v[op.in.right[vb]];
			}
			else { // elements is in B
				if (xj != cj) return 0.0;
				/* Vectorized index in A: (ci, bi) => va */
				const index_t va = ci + xi * rows_A;
				return v[op.in.left[va]];
			}
		}
		double partial(const index_t i, index_t j, index_t k) const {
			/*
			Grad[Grad[{{ a, c}, {b, d}}.{{ e, g}, {f, h}}, {a,b,c,d,e,f,g,h}], {a,b,c,d,e,f,g,h}]

			Grad[Grad[ { a e + c f, b e + d f, a g + c h, b g + d h }, {a,b,c,d,e,f,g,h} ], {a,b,c,d,e,f,g,h} ]

		(0)					(1)					(2)					(3)					(4)					(5)					(6)					(7)
{0,0,0,0,1,0,0,0}	{0,0,0,0,0,0,0,0}	{0,0,0,0,0,1,0,0}	{0,0,0,0,0,0,0,0}	{1,0,0,0,0,0,0,0}	{0,0,1,0,0,0,0,0}	{0,0,0,0,0,0,0,0}	{0,0,0,0,0,0,0,0}
{0,0,0,0,0,0,0,0}	{0,0,0,0,1,0,0,0}	{0,0,0,0,0,0,0,0}	{0,0,0,0,0,1,0,0}	{0,1,0,0,0,0,0,0}	{0,0,0,1,0,0,0,0}	{0,0,0,0,0,0,0,0}	{0,0,0,0,0,0,0,0}
{0,0,0,0,0,0,1,0}	{0,0,0,0,0,0,0,0}	{0,0,0,0,0,0,0,1}	{0,0,0,0,0,0,0,0}	{0,0,0,0,0,0,0,0}	{0,0,0,0,0,0,0,0}	{1,0,0,0,0,0,0,0}	{0,0,1,0,0,0,0,0}
{0,0,0,0,0,0,0,0}	{0,0,0,0,0,0,1,0}	{0,0,0,0,0,0,0,0}	{0,0,0,0,0,0,0,1}	{0,0,0,0,0,0,0,0}	{0,0,0,0,0,0,0,0}	{0,1,0,0,0,0,0,0}	{0,0,0,1,0,0,0,0}

			Outer row is i, the vectorized result element
			Outer column is j, the vectorized input element, with A on the left, B on the right side
			Inner column is k, the vectorized input element, idem

			Vectorized elements are
			 (0)  (1)  (2)  (3)  (4)  (5)  (6)  (7)
			[A00, A10, A01, A11, B00, B10, B01, B11]

			List coordinates of 1's for reference:

			(0,0,4) => d²C00/dA00²dB00²
			(0,2,5) => d²C00/dA01²dB10²
			(0,4,0) => d²C00/dB00²dA00²
			(0,5,3) => d²C00/dB10²dA01²

			(1,1,4) => d²C10/dA10²dB00²
			(1,3,5) => d²C10/dA11²dB10²
			(1,4,1) => d²C10/dB00²dA10²
			(1,5,3) => d²C10/dB10²dA11²

			(2,0,6) => d²C01/dA00²dB01²
			(2,2,7) => d²C01/dA01²dB11²
			(2,6,0) => d²C01/dB01²dA00²
			(2,7,2) => d²C01/dB11²dA01²

			(3,1,6) => d²C11/dA10²dB01²
			(3,3,7) => d²C11/dA11²dB11²
			(3,6,1) => d²C11/dB01²dA10²
			(3,7,3) => d²C11/dB11²dA11²

			Using indexing scheme A(ai, aj) B(bi, bj) C(ci cj), the Hessian is 1 iif:

			(ci = ai) and (cj = bj) and (aj = bi)

			=> In words, the Hessian is 1 iif
			-the result's row and left matrix's row match
			-the result's col and right matrix's col match
			-the left matrix's col and the right matrix's row match

			This is consistent with the 'structure' of matrix product, that is,
			element in result is the dot product of matching left matrix row (ci = ai)
			and matching right matrix column (cj = bj); and within a dot product the
			Hessian is 1 iif the element indices match in their respective vectors (aj = bi)
			*/

			const index_t rows_A = op.in.left.dim[0];
			const index_t cols_A = op.in.left.dim[1];
			const index_t rows_B = op.in.right.dim[0];
			const index_t rows_C = rows_A;
			const index_t size_A = rows_A * cols_A;

			/* First checkpoint: j and k must refer to elements of different matrices.
			Check that and order them so that j must refer to an element of A and k
			must refer to an element of B */
			if (j > k) std::swap(j, k);
			if (j >= size_A || k < size_A) return 0.;

			/* After this point, it is guaranteed that j refers to A and k refers to B */

			/* Map vectorized operand element index j onto matrix index (ai, aj) */
			const index_t ai = j % rows_A;
			const index_t aj = j / rows_A;

			/* Map vectorized operand element index k onto matrix index (bi, bj) */
			const index_t k_aligned = k - size_A;
			const index_t bi = k_aligned % rows_B;
			const index_t bj = k_aligned / rows_B;

			/* Map vectorized result element index i onto matrix index (ci, cj) */
			const index_t ci = i % rows_C;
			const index_t cj = i / rows_C;

			/* Check conditions for unit Hessian element */
			if (ci == ai && cj == bj && aj == bi) return 1.;
			else return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v };
	}
};


/* C = A.B, A is free, B is fixed */
template<> struct MultiplyMatrixMatrix<true, false> :
	OpIn::InTensor<2>, OpConst::ConstTensor<2>, OpOut::Range,
	HessianAlwaysZero, OpOutSize::Matrix_Matrix_LeftProductSize
{
	template< bool a, bool b> using family_t = MultiplyMatrixMatrix< a, b >;

	MultiplyMatrixMatrix(const OpIn::InTensor<2> in, const OpConst::ConstTensor<2> constant, const OpOut::Range out) :
		OpIn::InTensor<2>{ in }, OpConst::ConstTensor<2>{ constant }, OpOut::Range{ out }
	{
		assert(this->in.dim[1] /* cols(A) */ == this->constant.dim[0] /* rows(B) */ &&
			"Size mismatch between left and right operand");
	}
	void evaluate(std::vector< double >& v) const {

		const size_t a_nrows = in.dim[0];
		const size_t b_ncols = constant.dim[1];
		/* cols(A) = rows(B) */
		const size_t K = constant.dim[0];
		assert(K == in.dim[1]);

		/* ROW LOOP */
		for (size_t row = 0; row < a_nrows; row++) {
			/* COLUMN LOOP */
			for (size_t col = 0; col < b_ncols; col++) {
				double x = 0.;
				/* INNER LOOP */
				for (size_t k = 0; k < K; k++) {
					const double& a_ijk = v[in[row + k * a_nrows]];
					const double& b_ijk = constant[k + col * K];
					x += a_ijk * b_ijk;
				}
				v[out[0] + row + col * a_nrows] = x;
			}
		}
	}
	struct LocalDiff {
		const MultiplyMatrixMatrix<true, false>& op;
		const std::vector< double >& v;
		double partial(const index_t i, const index_t j) const {
			/*
			See details in MultiplyMatrixMatrix<true, true> above.

			Jacobian formula:
			dC_kl / dA_ij = B_jl if i==k, 0 otherwise
			dC_kl / dB_ij = A_ki if j==l, 0 otherwise

			For readability we use index names prefixed with matrix name,
			A(ai, aj) B(bi, bj) C(ci cj)

			Here, A is the free operand, B is the constant
			*/
			const index_t rows_A = op.in.dim[0];
			const index_t rows_B = op.constant.dim[0];
			const index_t rows_C = rows_A;
			/* Map vectorized operand element index j onto matrix index A(ai, aj) */
			const index_t ai = j % rows_A;
			const index_t aj = j / rows_A;
			/* Map vectorized result element index i onto matrix index C(ci, cj) */
			const index_t ci = i % rows_C;
			const index_t cj = i / rows_C;

			if (ai != ci) return 0.0;
			/* Vectorized index in B: (aj, cj) => vb */
			const index_t vb = aj + cj * rows_B;
			return op.constant[vb];
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			/*
			See details in MultiplyMatrixMatrix<true, true> above.

			Here, j and k must refer to the same operand A, hence
			the Hessian is always 0 */
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v };
	}
};

/* C = A.B, A is fixed, B is free */
template<> struct MultiplyMatrixMatrix<false, true> :
	OpIn::InTensor<2>, OpConst::ConstTensor<2>, OpOut::Range,
	HessianAlwaysZero, OpOutSize::Matrix_Matrix_RightProductSize
{
	template< bool a, bool b> using family_t = MultiplyMatrixMatrix< a, b >;

	MultiplyMatrixMatrix(const OpIn::InTensor<2> in, const OpConst::ConstTensor<2> constant, const OpOut::Range out) :
		OpIn::InTensor<2>{ in }, OpConst::ConstTensor<2>{ constant }, OpOut::Range{ out }
	{
		assert(this->constant.dim[1] /* A */ == this->in.dim[0] /* B */ && "Size mismatch between left and right operand");
	}
	void evaluate(std::vector< double >& v) const {
		/* A is the constant, B is the operand */
		const size_t a_nrows = constant.dim[0];
		const size_t b_ncols = in.dim[1];
		/* cols(A) = rows(B) */
		const size_t K = in.dim[0];
		assert(K == constant.dim[1]);

		/* ROW LOOP */
		for (size_t row = 0; row < a_nrows; row++) {
			/* COLUMN LOOP */
			for (size_t col = 0; col < b_ncols; col++) {
				double x = 0.;
				/* INNER LOOP */
				for (size_t k = 0; k < K; k++) {
					const double& a_ijk = constant[row + k * a_nrows];
					const double& b_ijk = v[in[k + col * K]];
					x += a_ijk * b_ijk;
				}
				v[out[0] + row + col * a_nrows] = x;
			}
		}
	}
	struct LocalDiff {
		const MultiplyMatrixMatrix<false, true>& op;
		const std::vector< double >& v;
		double partial(const index_t i, const index_t j) const {
			/*
			See details in MultiplyMatrixMatrix<true, true> above.

			Jacobian formula:
			dC_kl / dA_ij = B_jl if i==k, 0 otherwise
			dC_kl / dB_ij = A_ki if j==l, 0 otherwise

			For readability we use index names prefixed with matrix name,
			A(ai, aj) B(bi, bj) C(ci cj)

			Here, A is the constant, B is the input operand
			*/
			const index_t rows_A = op.constant.dim[0];
			const index_t rows_B = op.in.dim[0];
			const index_t rows_C = rows_A;
			/* Map vectorized operand element index j onto matrix index B(bi, bj) */
			const index_t bi = j % rows_B;
			const index_t bj = j / rows_B;
			/* Map vectorized result element index i onto matrix index C(ci, cj) */
			const index_t ci = i % rows_C;
			const index_t cj = i / rows_C;

			if (bj != cj) return 0.0;
			/* Vectorized index in A: (ci, bi) => va */
			const index_t va = ci + bi * rows_A;
			return op.constant[va];
		}
		constexpr double partial(const index_t i, const index_t j, const index_t k) const {
			/*
			See details in MultiplyMatrixMatrix<true, true> above.

			Here, j and k must refer to the same operand A, hence
			the Hessian is always 0 */
			return 0.;
		}
	};
	LocalDiff local_diff(const std::vector< double >& v) const {
		return LocalDiff{ *this, v };
	}
};

/****************************************************************************
*** OPERATOR DECLARATION
*****************************************************************************/

using op_multiply_types = std::tuple<
	MultiplyScalarScalar< true, true >, MultiplyScalarScalar< true, false >, MultiplyScalarScalar< false, true >,
	MultiplyVectorScalar< true, true >, MultiplyVectorScalar< true, false >, MultiplyVectorScalar< false, true >,
	MultiplyVectorVector< true, true >, MultiplyVectorVector< true, false >, MultiplyVectorVector< false, true >,
	MultiplyMatrixMatrix< true, true >, MultiplyMatrixMatrix< true, false >, MultiplyMatrixMatrix< false, true >
>;