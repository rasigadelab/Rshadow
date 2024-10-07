/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include <Eigen/SparseCore>
#include "sparse_matrix.h"

/* Eigen sparse matrix */
struct EigenSparseMat : Eigen::SparseMatrix< double, Eigen::ColMajor, Eigen::Index >
{
	/* Constructor inheritance */
	using Eigen::SparseMatrix< double, Eigen::ColMajor, Eigen::Index >::SparseMatrix;

	/* Construct from dynamic map-based sparse matrix. Dimensions
	in optional argument fixed_indices are neutralized
	(diagonal -1, off-diagonal 0). */
	EigenSparseMat(
		const SparseSymMat& hessian_map,
		const std::vector< index_t > fixed_indices = {});

	/* Negate all matrix elements. */
	EigenSparseMat& negate();
};