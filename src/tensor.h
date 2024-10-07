/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once

#include <vector>
#include "utilities.h"

/* Tensor dimension descriptor */
class TensorDim {
private:
	static size_t product(const std::vector< size_t > vec) {
		if (vec.size() == 0) return 0;
		size_t n = 1;
		for (const auto& ndim : vec) {
			n *= ndim;
		}
		return n;
	}
public:
	/* Vector dimension sizes. dim.size() is the number of dimensions.
	dim[i] in the number of slices along the ith dimension. Examples: 
	{1} is a scalar; {3,3} is a square matrix with 3 rows and 3 columns. */
	std::vector< size_t > dim;

	TensorDim(const std::vector< size_t > dim) : dim{ dim } {
		for (size_t i = 0; i < dim.size(); i++) {
			assert(dim[i] > 0);
		}
	}

	/* Number of coefficients equals the product of the number of slices
	across each dimension. */
	size_t size() const {
		return TensorDim::product(dim);
	}
	bool is_null() const {
		return this->size() == 0;
	}
	bool is_scalar() const {
		return this->size() == 1;
	}
	bool is_vector() const {
		/* A K-tensor is a vector iif at least K-1 dimensions have size 1.
		Equivalently, a tensor is a vector iif there is a dimension whose size
		equals the tensor size. */
		const size_t tensor_size = size();
		for (const size_t& dim_size : dim) {
			if (dim_size == tensor_size) return true;
			if (dim_size > 1) return false;
		}
		return false;
	}
	bool is_matrix() const {
		return dim.size() == 2;
	}
	bool operator==(const TensorDim& x) const {
		/* Using == overload for vectors in vector_overloads.h */
		return dim == x.dim;
	}

	/*** INDEX METHODS ***********************************/

	/* Map 2D index pair (i,j) onto vectorized index */
	index_t vec_index(const index_t i, const index_t j) const {
		const index_t vi = i + j * dim[0];
		return vi;
	}
	/* Map 3D index triple (i,j,k) onto vectorized index */
	index_t vec_index(const index_t i, const index_t j, const index_t k) const {
		const index_t vi = i + j * dim[0] + k * dim[1] * dim[0];
		return vi;
}
	/* Map arbitrary tensor index tuple {i_1, i_2, ..., i_K} onto vectorized index */
	index_t vec_index(const std::vector< index_t >& ivec) const {
		assert(ivec.size() <= dim.size());
		index_t vi = 0;
		index_t n_ = 1; // Dimension
		for (size_t k = 0; k < ivec.size(); k++) {
			vi += ivec[k] * n_;
			n_ *= dim[k];
		}
		return vi;
	}

	/*** DIMENSION SIZES ***********************************/

	/* Number of matrix rows (Tensor must be a matrix) */
	size_t nrow() const {
		assert(is_matrix());
		return dim[0];
	}
	/* Number of matrix columns (Tensor must be a matrix) */
	size_t ncol() const {
		assert(is_matrix());
		return dim[1];
	}
};

/* R-like tensor with column-major storage and
dimension vector */
struct Tensor : TensorDim {
	std::vector< double > val;
	Tensor(const TensorDim& dim) : TensorDim{ dim } {
		val.resize(size());
	}
	Tensor(const std::vector< size_t >& dimvec = { 1 }) : Tensor{ TensorDim{ dimvec } } {}
	Tensor(const double& x) : TensorDim{ { 1 } }, val{ { x } } {}
	Tensor(const std::vector< double >& x) : TensorDim{ { x.size() } }, val{ x } {}
	Tensor(const std::vector< double >&& x) : TensorDim{ { x.size() } }, val{ x } {}
	Tensor(const std::vector< double >&& x, const std::vector< size_t >&& dim) : TensorDim{ dim }, val{ x } {
		assert(val.size() == size());
	}
	Tensor(const std::vector< double >& x, const std::vector< size_t >& dim) : TensorDim{ dim }, val{ x } {
		assert(val.size() == size());
	}
	/* Scalar conversion */
	double& scalar() {
		assert(is_scalar());
		return val[0];
	}
	double scalar() const {
		assert(is_scalar());
		return val[0];
	}
	/* Vector conversion */
	std::vector< double >& vector() {
		assert(is_vector());
		return val;
	}
	std::vector< double > vector() const {
		assert(is_vector());
		return std::vector< double >(val);
	}
	/* Underlying data pointer (assuming you know what you're doing...) */
	double* data() {
		return val.data();
	}
	const double* data() const {
		return val.data();
	}

	/*** ACCESS TO TENSOR COEFFICIENTS *************************/

	/* Check that a vectorized index is valid for the tensor */
	bool vec_index_is_valid(const index_t vi) const {
		return vi >= 0 && vi < index_t(val.size());
	}
	/* Bracket access */
	double& operator[](const index_t vi) {
		assert(vec_index_is_valid(vi));
		return val[vi];
	}
	double operator[](const index_t vi) const {
		assert(vec_index_is_valid(vi));
		return val[vi];
	}
	/* Vector access */
	double& operator()(const index_t vi) {		
		return operator[](vi);
	}
	double operator()(const index_t vi) const {
		return operator[](vi);
	}
	/* Matrix access */
	double& operator()(const index_t i, const index_t j) {
		const index_t vi = vec_index(i, j);
		return operator[](vi);
	}
	double operator()(const index_t i, const index_t j) const {
		const index_t vi = vec_index(i, j);
		return operator[](vi);
	}
	/* 3D tensor access */
	double& operator()(const index_t i, const index_t j, const index_t k) {
		const index_t vi = vec_index(i, j, k);
		return operator[](vi);
	}
	double operator()(const index_t i, const index_t j, const index_t k) const {
		const index_t vi = vec_index(i, j, k);
		return operator[](vi);
	}
	/* Arbitrary tensor access */
	double& operator()(const std::vector< index_t >& ivec) {
		const index_t vi = vec_index(ivec);
		return operator[](vi);
	}
	double operator()(const std::vector< index_t >& ivec) const {
		const index_t vi = vec_index(ivec);
		return operator[](vi);
	}

	/*** MANIPULATION *********************************************/

	/* Fill tensor with a scalar value */
	Tensor& fill(const double x) {
		for (double& v : val) v = x;
		return *this;
	}
	/* Make a basic vector an explicit column vector */
	Tensor& make_col_vector() {
		assert(is_vector());
		dim.resize(2);
		dim[0] = val.size();
		dim[1] = 1;
		return *this;
	}
	/* Make a basic vector an explicit row vector */
	Tensor& make_row_vector() {
		assert(is_vector());
		dim.resize(2);
		dim[0] = 1;
		dim[1] = val.size();
		return *this;
	}
};

