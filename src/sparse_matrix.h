/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once

#include <vector> 
#include <map> 

#include "utilities.h"

/****************************************************************************
*** SPARSE SYMMETRIC MATRIX
*****************************************************************************/

/* A std::map-based dynamic sparse symmetric matrix */
class SparseSymMat {
private:
	/* Number of rows or columns */
	size_t width_ = 0;
public:

	using row_t = std::map< index_t, double >;
	using matrix_t = std::map< index_t, row_t >;
	matrix_t matrix;

	/* FIXME this should return nrow * ncol */
	size_t width() const {
		return width_;
	}
	SparseSymMat& set_width(const size_t sz) {
		width_ = sz;
		return *this;
	}

	SparseSymMat(size_t width) : width_{ width } {}

	size_t nrow() const { return width_; }
	size_t ncol() const { return width_; }
	size_t size() const { return width_ * width_; }

	/* Read-only matrix value */
	double read(const index_t i, const index_t j) const {
		auto it_i = matrix.find(i);
		if (it_i == matrix.end()) return 0.0;
		auto it_j = it_i->second.find(j);
		if (it_j == it_i->second.end()) return 0.0;
		return it_j->second;
	}

	/* Pointer to a matrix row, nullptr if row is empty */
	row_t* get_row_ptr(const index_t i) {
		auto it_i = matrix.find(i);
		if (it_i == matrix.end()) return nullptr;
		return &it_i->second;
	}

	/* Erase the ith row and the ith column */
	void erase(const index_t i) {
		auto row_i = matrix.find(i);
		if (row_i == matrix.end()) return;

		for (const auto& it_j : row_i->second) {
			const index_t j = it_j.first;
			if (j == i) continue; // Skip because we erase row i after the loop anyway

			auto row_j = matrix.find(it_j.first);
			assert(row_j != matrix.end());

			row_j->second.erase(i);
			if (row_j->second.size() == 0) {
				matrix.erase(row_j);
			}
		}
		matrix.erase(row_i);
	}

	/* Opaque object referencing an entry of the Hessian, with read 
	and write ability to manage	the 2-way entry scheme consistently */
	class elem_ref_t {
	private:
		index_t i;
		index_t j;
		matrix_t& matrix;
		/* Iterator to row i */
		matrix_t::iterator i_;
		/* Iterator to element (i,j) */
		row_t::iterator ij;
		/* Iterator to row j */
		matrix_t::iterator j_;
		/* Iterator to element (j,i) */
		row_t::iterator ji;
		/* Element iterator is valid? */
		bool is_valid_ij = false;

	public:
		elem_ref_t(matrix_t& matrix, const index_t i, const index_t j) :
			matrix{ matrix }, i{ i }, j{ j } {};

		/* Retrieve or insert the element */
		void prepare_write() {
			/* Row iterator, either new or existing */
			i_ = matrix.insert({ i, row_t{} }).first;
			/* Element iterator, either new or existing */
			ij = i_->second.insert({ j, 0.0 }).first;

			if (j != i) {
				/* Row iterator, either new or existing */
				j_ = matrix.insert({ j, row_t{} }).first;
				/* Element iterator, either new or existing */
				ji = j_->second.insert({ i, 0.0 }).first;
			}
			is_valid_ij = true;
		}
		/* Delete the element, practically setting to zero. Also remove row or column if needed. */
		void set_zero() {
			if (is_valid_ij == false) return;

			i_->second.erase(ij);
			if (i_->second.size() == 0) matrix.erase(i_);

			is_valid_ij = false;

			if (j == i) return;

			j_->second.erase(ji);
			if (j_->second.size() == 0) matrix.erase(j_);
		}

		void operator+=(const double x) {
			if (x == 0.0) return;
			if (is_valid_ij == false) prepare_write();
			ij->second += x;
			if (j == i) return;
			ji->second += x;
		}

		void operator=(const double x) {
			if (x == 0.0) return set_zero();
			if (is_valid_ij == false) prepare_write();
			ij->second = x;
			if (j == i) return;
			ji->second = x;
		}

	};

	/* Return a reference to the element at position (i, j) */
	elem_ref_t get(const index_t i, const index_t j) {
		return { this->matrix, i, j };
	}

	/* Print matrix to console in dense form */
	void print() const;
};
