/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include <Eigen/Core>
#include "tensor.h"

class MatrixMap : public Eigen::Map<Eigen::MatrixXd> {
public:
	using Eigen::Map<Eigen::MatrixXd>::Map;
	MatrixMap(Tensor& x) :
		Map(x.val.data(), x.nrow(), x.ncol())
	{
		assert(x.dim.size() == 2);
	}
	Eigen::Map<Eigen::MatrixXd>& eigen() {
		return static_cast<Eigen::Map<Eigen::MatrixXd>&>(*this);
	}
};

class ConstMatrixMap : public Eigen::Map<const Eigen::MatrixXd> {
public:
	using Eigen::Map<const Eigen::MatrixXd>::Map;
	ConstMatrixMap(const Tensor& x) :
		Map(x.val.data(), x.nrow(), x.ncol())
	{
		assert(x.dim.size() == 2);
	}
	Eigen::Map<const Eigen::MatrixXd>& eigen() {
		return static_cast<Eigen::Map<const Eigen::MatrixXd>&>(*this);
	}
};