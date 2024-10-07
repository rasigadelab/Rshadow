/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "tape.h"
#include "trace.h"

void Tape::write_tensor_map_to_trace(Trace& trace, const TensorMap& map) const {
	for (const auto& it : to_tensor_map) {
		for (size_t i = 0; i < map[it.second].tensor().val.size(); i++) {
			trace[it.first + i] = map[it.second].tensor().val[i];
		}
	}
}

void Tape::write_trace_to_tensor_map(const Trace& trace, TensorMap& map) const {
	for (const auto& it : to_tensor_map) {
		for (size_t i = 0; i < map[it.second].tensor().val.size(); i++) {
			map[it.second].tensor().val[i] = trace[it.first + i];
		}
	}
}

/* Get a Trace object with correct size and mapped variables written */
Trace Tape::get_trace(const TensorMap& map) {
	assert(n_trace_size >= n_input_size);
	Trace trace(*this);
	write_tensor_map_to_trace(trace, map);
	return trace;
}