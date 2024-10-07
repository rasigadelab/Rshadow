/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "convert.h"

Eigen::MatrixXd Convert::MatrixXd_from_SparseSymMat(const SparseSymMat& mat) {
	using sparse_mat_t = Eigen::SparseMatrix< double, Eigen::ColMajor, Eigen::Index >;

	/* Convert triplet list to Eigen form */
	std::vector< Eigen::Triplet< double, Eigen::Index > > triplet_list;
	triplet_list.reserve(mat.matrix.size() * 2);

	for (const auto& row : mat.matrix) {
		const Eigen::Index i = Eigen::Index(row.first);
		for (const auto& entry : row.second) {
			const Eigen::Index j = Eigen::Index(entry.first);
			/* negate because Chol wants positive definite mat
			while Hessian is negative negative (*should be* negative definite) */
			triplet_list.push_back({ i, j, -entry.second });
		}
	}

	sparse_mat_t Eigen_hessian(mat.width(), mat.width());
	Eigen_hessian.setFromTriplets(triplet_list.begin(), triplet_list.end());
	Eigen_hessian.makeCompressed();

	Eigen::MatrixXd Eigen_mat(mat.width(), mat.width());
	Eigen_mat.setZero();

	for (const auto& row : mat.matrix) {
		const Eigen::Index i = Eigen::Index(row.first);
		for (const auto& entry : row.second) {
			const Eigen::Index j = Eigen::Index(entry.first);

			Eigen_mat(i, j) = entry.second;

		}
	}

	return Eigen_mat;
}

Tensor Convert::Tensor_from_MatrixXd(const Eigen::MatrixXd& Eigen_mat) {
	Tensor tensor( TensorDim({ size_t(Eigen_mat.rows()), size_t(Eigen_mat.cols()) }));
	for (size_t i = 0; i < size_t(Eigen_mat.size()); i++) {
		tensor.val[i] = Eigen_mat.data()[i];
	}
	return tensor;
}

Tensor Convert::Tensor_from_SparseSymMat(const SparseSymMat& mat) {

	Eigen::MatrixXd Eigen_mat = MatrixXd_from_SparseSymMat(mat);

	return Tensor_from_MatrixXd(Eigen_mat);
}