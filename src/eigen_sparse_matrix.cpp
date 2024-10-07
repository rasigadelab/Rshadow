/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "eigen_sparse_matrix.h"

/* Construct from dynamic map-based sparse matrix. Dimensions
in optional argument fixed_indices are neutralized
(diagonal -1, off-diagonal 0). */
EigenSparseMat::EigenSparseMat(
	const SparseSymMat& hessian_map,
	const std::vector< index_t > fixed_indices) :
	SparseMatrix(hessian_map.width(), hessian_map.width())
{
	/* Convert triplet list to Eigen form */
	std::vector< Eigen::Triplet< double, Eigen::Index > > triplet_list;
	triplet_list.reserve(hessian_map.matrix.size() * 2);

	for (const auto& row : hessian_map.matrix) {

		/* Fixed values, push diagonal entry and skip rest of the row */
		bool skip = false;
		for (index_t fixed_i : fixed_indices) {
			if (row.first == fixed_i) {
				triplet_list.push_back({ row.first, row.first, -1. });
				skip = true;
			}
		}
		if (skip) continue;

		const Eigen::Index i = Eigen::Index(row.first);
		for (const auto& entry : row.second) {
			const Eigen::Index j = Eigen::Index(entry.first);
			triplet_list.push_back({ i, j, entry.second });
		}
	}
	this->setFromTriplets(triplet_list.begin(), triplet_list.end());
	this->makeCompressed(); // Compression mandatory before using solvers
}

EigenSparseMat& EigenSparseMat::negate() {
	for (Eigen::Index k = 0; k < this->outerSize(); k++) {
		for (EigenSparseMat::InnerIterator it(*this, k); it; ++it) {
			it.valueRef() *= -1.;
		}
	}
	return *this;
}