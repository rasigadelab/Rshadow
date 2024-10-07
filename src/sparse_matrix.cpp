/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include <iostream>

#include "sparse_matrix.h"
#include "convert.h"

void SparseSymMat::print() const {

	Eigen::MatrixXd mat = Convert::MatrixXd_from_SparseSymMat(*this);

	std::cout << mat << std::endl;
}