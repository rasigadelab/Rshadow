/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "tensor.h"
#include "sparse_matrix.h"

struct Convert
{
	static Tensor Tensor_from_MatrixXd(const Eigen::MatrixXd&);
	static Eigen::MatrixXd MatrixXd_from_SparseSymMat(const SparseSymMat&);
	static Tensor Tensor_from_SparseSymMat(const SparseSymMat&);
};

