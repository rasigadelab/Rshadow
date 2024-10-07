/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "solver.h"

#include <Eigen/Core>
#include <Eigen/SparseLU>

#include <iostream>

#include "utilities.h"
#include "brent_optimize.h"
#include "eigen_sparse_matrix.h"



/* Objective function to find the optimal step size along the
   Newton direction vector. */
struct BrentOptimizeObjectiveFunctor {
	Solver& solver;
	BrentOptimizeObjectiveFunctor(Solver& solver) : solver{ solver } {}

	size_t n_eval = 0;

	double operator()(double step) {

		n_eval++;

		/* Forward-evaluate the tape */
		for (size_t i = 0; i < solver.trace.tape.input_size(); i++) {
			solver.trace.values[i] = solver.param_buffer[i] + step * solver.direction_buffer[i];
		}
		double objective = solver.trace.play_forward().result();
		
		if (std::isfinite(objective)) return objective;
		
		if (objective > std::numeric_limits< double >::max()) {
			throw std::logic_error("Infinitely high objective encountered.");
		}

		return -std::numeric_limits< double >::infinity();
	}
};

/* Safeguarded Newton-Marquardt optimization */
Solver& Solver::maximize() {

	/* Problem size */
	const size_t N = trace.tape.input_size();
	/* Objective function values */
	double objective_old = -std::numeric_limits< double >::infinity();
	double objective_new = trace.play().result();
	n_eval_forward++;
	n_eval_reverse++;

	/* Objective functor for Brent optimization */
	BrentOptimizeObjectiveFunctor brent_functor(*this);

	/* Prepare Hessian in Eigen form and sparse LU solver */
	EigenSparseMat Eigen_hessian(trace.hessian, fixed_parameter_indices);
	Eigen::SparseLU< decltype(Eigen_hessian), Eigen::COLAMDOrdering<Eigen::Index> > LU_solver;
	LU_solver.analyzePattern(Eigen_hessian); // Value-independent so can be done before the loop
	hessian_buffer.resize(Eigen_hessian.nonZeros());	

	/* Map buffers to Eigen maps: */

	/* Parameters in Eigen form */
	Eigen::Map< Eigen::VectorXd > Eigen_parameters(trace.values.data(), N);
	/* Keep track of original parameters in the param_buffer field*/
	Eigen::Map< Eigen::VectorXd > Eigen_orig_params(param_buffer.data(), N);
	/* Gradient in Eigen form */
	Eigen::Map< Eigen::VectorXd > Eigen_gradient(trace.adjoints.data(), N);
	/* Newton direction, -f(x)/f'(x), in our case, -Hessian^-1 x Gradient */
	Eigen::Map< Eigen::VectorXd > Eigen_newton_direction(direction_buffer.data(), N);

	/*********************************************************************************************/
	/* NEWTON ITERATIONS *************************************************************************/
	/*********************************************************************************************/

	size_t newton_step_index = 0;
	while ((objective_new - objective_old > config.objective_tolerance) && (newton_step_index++ <= config.max_iterations)) {

		/* Reset objective evaluation counter and keep track of current parameter value*/
		brent_functor.n_eval = 0;
		Eigen_orig_params = Eigen_parameters;

		/*** REGULARIZED NEWTON DIRECTION *********************************************************/

		/* Fixed parameters (eg for profiling) have their derivatives set to zero.
		NB this is handled directly by EigenSparseMat for second derivatives */
		for (index_t fixed_index : fixed_parameter_indices) {
			trace.adjoints[fixed_index] = 0.;
		}
		
		/* We use the same matrix and its diagonal repeatedly. Accessing sparse elements requires
		binary search so it's a good idea to keep pointers to the elements once. TODO if we have
		some stability guarantees on the Hessian, this may be moved out of the loop */
		/* https://stackoverflow.com/questions/38474242/eigen-diagonal-update-of-sparse-matrix */

		/* Hessian in Eigen form, and a vector of pointers to the diagonal elements */
		Eigen_hessian = EigenSparseMat(trace.hessian, fixed_parameter_indices);

		/* Attempt an unregularized solution */
		size_t n_regul = 0;  // No. of regularization attempts
		double lambda = 0.0; // Regularization strength, in [0,1]
		LU_solver.factorize(Eigen_hessian);
		if (LU_solver.info() != Eigen::Success) n_regul++;
		else {
			Eigen_newton_direction = -LU_solver.solve(Eigen_gradient);
			if (LU_solver.info() != Eigen::Success) n_regul++;
		}		

		if (n_regul > 0) {
			/* Regularize until gradient */

			/* Store the current values in the Hessian buffer */
			size_t hbuf_pos = 0;
			for (int k = 0; k < Eigen_hessian.outerSize(); ++k)
				for (EigenSparseMat::InnerIterator it(Eigen_hessian, k); it; ++it)
				{
					hessian_buffer[hbuf_pos++] = it.value();
				}

			/* Increase per regularization attempt */
			double reg_step_size = 1. / config.max_regularization_attempts;

			/* No. of attempts */
			while (n_regul <= config.max_regularization_attempts) {

				lambda = std::pow(n_regul * reg_step_size, config.regularization_damping_factor);

				const double I_weight = lambda;
				const double H_weight = 1. - I_weight;

				assert(I_weight <= 1.0);

				/* Weighted combination of (1-lambda) H + lambda I */
				hbuf_pos = 0;
				for (int k = 0; k < Eigen_hessian.outerSize(); ++k)
					for (EigenSparseMat::InnerIterator it(Eigen_hessian, k); it; ++it)
					{
						/* (1-lambda) H + 0, for off diagonal elements */
						it.valueRef() = hessian_buffer[hbuf_pos++] * H_weight;
						/* Add lambda to diagonal elements */
						if (it.row() == it.col()) it.valueRef() += I_weight;
					}

				LU_solver.factorize(Eigen_hessian);
				if (LU_solver.info() != Eigen::Success) n_regul++;
				else {
					Eigen_newton_direction = -LU_solver.solve(Eigen_gradient);
					if (LU_solver.info() != Eigen::Success) n_regul++;
				}
			}
		}

		/*** BACKTRACKING LINE SEARCH ******************************************************/

		/* Define a feasible search interval around the current point */
		double brent_left = config.brent_boundary_left;
		double brent_right = config.brent_boundary_right;

		/* Restrict Brent interval to feasible area */
		while (!std::isfinite(brent_functor(brent_left))) {
			brent_left *= config.brent_feasible_search_restriction_factor;
		}
		while (!std::isfinite(brent_functor(brent_right))) {
			brent_right *= config.brent_feasible_search_restriction_factor;
		}

		/* In general, the Brent tolerance should not be tighter than the general
		 objective tolerance. However, if the tolerance is larger than the search interval,
		 the objective may decrease more than what is admissible. So the effective tolerance
		 is set to the smaller value of the general tolerance, or the squared interval size. */
		
		const double brent_width = brent_right - brent_left;	
		const double brent_tol = fmin(
			config.objective_tolerance * config.brent_tolerance_factor,
			brent_width * brent_width
		);
		constexpr bool MAXIMIZE = true; // For readability

		brent_opt_output_t brent_out = brent_optimize(
			brent_functor,
			brent_left, brent_right,
			MAXIMIZE, brent_tol
		);

		/* Did we decrease the objective ? */
		if (brent_out.objective < objective_new - brent_tol) {
			throw std::logic_error("Failure of backtracking line search.");
		}

		objective_old = objective_new;
		objective_new = brent_out.objective;
		trace.play_reverse();

		n_eval_forward += brent_functor.n_eval;
		n_eval_reverse++;

		if (config.diagnostic_mode == true) {
			/* Store gradient and Hessian for diagnostics */
			std::vector< double > gradient_state(N);
			Eigen::Map< Eigen::VectorXd > Eigen_gradient_state(gradient_state.data(), N);
			Eigen_gradient_state = Eigen_gradient;

			std::vector< double > hessian_state(N * N);
			Eigen::Map< Eigen::MatrixXd > Eigen_hessian_state(hessian_state.data(), N, N);
			Eigen_hessian_state = Eigen::MatrixXd(Eigen_hessian);

			SolverState state{
				newton_step_index, objective_old, objective_new, lambda,
				{ param_buffer }, { gradient_state }, { hessian_state }, { direction_buffer },
				brent_left, brent_right, brent_out.min, brent_functor.n_eval, n_regul
			};
			/* Odd enough, we can't rely on the default struct constructor as it silently
			ignores the last fields, at least apparently. Let's hope that it's not a more 
			severe problem (like mem overwrite due to Eigen or other subtelties ?)*/
			state.n_regul = n_regul;
			assert(state.n_regul == n_regul);
			assert(state.n_eval = brent_functor.n_eval);

			states.push_back(state);
		}

	} // END while objective increases
	return *this;
}

SolverState& SolverState::print() {

	const size_t N = parameters.size();

	const Eigen::Map< Eigen::VectorXd > Eigen_params(parameters.data(), N);
	const Eigen::Map< Eigen::VectorXd > Eigen_gradient(gradient.data(), N);
	const Eigen::Map< Eigen::MatrixXd > Eigen_hessian(hessian.data(), N, N);
	const Eigen::Map< Eigen::VectorXd > Eigen_direction(direction.data(), N);

	std::cout << "Step #" << iter << ":" << std::endl;
	std::cout << "Parameter vector:\n" << Eigen_params << std::endl;
	std::cout << "Gradient vector:\n" << Eigen_gradient << std::endl;
	std::cout << "Hessian matrix:\n" << Eigen_hessian << std::endl;
	std::cout << "Regularization lambda = " << lambda << " found after " << n_regul << " regularization attempts. " << std::endl;
	std::cout << "Direction vector:\n" << Eigen_direction << std::endl;
	std::cout << "Optimal step amplitude = " << optstep << " found after " << n_eval << " objective evaluations. " << std::endl;
	std::cout << "Objective changed from " << objective_initial << " to " << objective_final << std::endl;
	std::cout << std::endl;
	
	return *this;
}