/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "trace.h"
#include "tape.h"
#include "spy.h"

Trace& Trace::play_forward() {
	this->tape.play_forward(*this);
	return *this;
}

Trace& Trace::play_reverse() {
	this->tape.play_reverse(*this);
	return *this;
}

Trace& Trace::play() {
	this->tape.play(*this);
	return *this;
}

double Trace::read_scalar(const Spy& spy) const {
	return values[spy.tape_begin()];
}

std::vector< double > Trace::read(const Spy& spy) const {
	std::vector< double > res(spy.size());
	std::copy(values.cbegin() + spy.tape_begin(), values.cbegin() + spy.tape_end(), res.begin());
	return res;
}

Tensor Trace::read_tensor(const Spy& spy) const {
	return Tensor( read(spy), spy.tensor().dim );
}