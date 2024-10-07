/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/

#include "s4_io.h"

template<> Rcpp::S4 to_S4<Tensor>(Tensor& x) {
	Tensor* ptr = new Tensor(x.val, x.dim);
	return wrap_ptr_S4(ptr);
};

template<> Rcpp::S4 to_S4<Spy>(Spy& x) {
	Spy* ptr = new Spy(static_cast<Tensor&>(x),
		x.tape, x.tape_begin());
	return wrap_ptr_S4(ptr);
};

#define TO_S4_CONFIG_SLOT(slot_name) \
config_S4.slot(#slot_name) = config_Cpp.slot_name;

template<> Rcpp::S4 to_S4< SolverConfig >(SolverConfig& config_Cpp) {
	Rcpp::S4 config_S4("shadow_solver_config");
	TO_S4_CONFIG_SLOT(max_iterations)
		TO_S4_CONFIG_SLOT(objective_tolerance)
		TO_S4_CONFIG_SLOT(diagnostic_mode)
		TO_S4_CONFIG_SLOT(max_regularization_attempts)
		TO_S4_CONFIG_SLOT(regularization_damping_factor)
		TO_S4_CONFIG_SLOT(brent_tolerance_factor)
		TO_S4_CONFIG_SLOT(brent_boundary_left)
		TO_S4_CONFIG_SLOT(brent_boundary_right)
		TO_S4_CONFIG_SLOT(brent_feasible_search_restriction_factor)
		return config_S4;
}
#undef TO_S4_CONFIG_SLOT


#define TO_S4_STATE_SLOT(slot_name) \
state_S4.slot(#slot_name) = state_Cpp.slot_name;

template<> Rcpp::S4 to_S4<SolverState>(SolverState& state_Cpp) {
	Rcpp::S4 state_S4("shadow_solver_state");
	TO_S4_STATE_SLOT(iter)
		TO_S4_STATE_SLOT(objective_initial)
		TO_S4_STATE_SLOT(objective_final)
		TO_S4_STATE_SLOT(lambda)
		TO_S4_STATE_SLOT(parameters)
		TO_S4_STATE_SLOT(gradient)
		TO_S4_STATE_SLOT(direction)
		TO_S4_STATE_SLOT(brent_left)
		TO_S4_STATE_SLOT(brent_right)
		TO_S4_STATE_SLOT(optstep)
		TO_S4_STATE_SLOT(n_eval)
		TO_S4_STATE_SLOT(n_solves)
		TO_S4_STATE_SLOT(n_regul)

		/* Make Hessian a proper matrix */
		size_t n_inputs = state_Cpp.parameters.size();
	Tensor hessian_tensor(state_Cpp.hessian, { n_inputs, n_inputs });
	state_S4.slot("hessian") = NumericVector_from_Tensor(hessian_tensor);

	return state_S4;
}
#undef TO_S4_STATE_SLOT


/* Convert tensor to R numeric format. By default, scalar/vector tensors
  are converted to R vectors without a 'dim' attribute. To always convert
  the dimension vector of the tensor, set force_dim_attr = true. */
Rcpp::NumericVector NumericVector_from_Tensor(const Tensor& tensor, bool force_dim_attr) {
	Rcpp::NumericVector xx = Rcpp::wrap(tensor.val);
	/* Push dimension attribute only for matrices and arrays */
	if (force_dim_attr || tensor.dim.size() > 1) {
		Rcpp::IntegerVector dim = Rcpp::wrap(tensor.dim);
		xx.attr("dim") = dim;
	}
	return xx;
}