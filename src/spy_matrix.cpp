/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "spy_matrix.h"
#include "eigen_matrixmap.h"

Tensor matmult(const Tensor& a, const Tensor& b) {
	ConstMatrixMap a_map(a);
	ConstMatrixMap b_map(b);
	Tensor c{ TensorDim({a.dim[0], b.dim[1]}) };
	MatrixMap c_map(c);
	c_map.eigen() = a_map.eigen() * b_map.eigen();
	return c;
}

Spy matmult(const Spy& a, const Spy& b) {
	assert(&a.tape == &b.tape);
	index_t out = a.tape.rec< MultiplyMatrixMatrix< true, true > >({ a, b });
	return Spy(matmult(Tensor(a), Tensor(b)), a.tape, out);
}

Spy matmult(const Spy& a, const Tensor& b) {
	index_t out = a.tape.rec< MultiplyMatrixMatrix< true, false > >(a, b);
	return Spy(matmult(Tensor(a), Tensor(b)), a.tape, out);
}

Spy matmult(const Tensor& a, const Spy& b) {
	index_t out = b.tape.rec< MultiplyMatrixMatrix< false, true > >(b, a);
	return Spy(matmult(Tensor(a), Tensor(b)), b.tape, out);
}