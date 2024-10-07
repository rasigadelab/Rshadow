/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once

#include "trace.h"
#include <functional>

/* A set of configuration parameters governing the optimization step */
struct SolverConfig {
	/* Maximum number of Newton step iterations */
	size_t max_iterations = 1000;
	/* Stopping criterion, difference of two consecutive objective values */
	double objective_tolerance = 1e-3;
	/* Keep track of details of optimization steps (SLOW!) */
	bool diagnostic_mode = false;
	/* Maximum of Tikhonov regularization attempts before falling back to gradient */
	size_t max_regularization_attempts = 10;
	/* Higher values lead to smaller initial regularization attempts, leading to more
	precise Newton steps but possibly more attempts before numerical stability is reached. */
	double regularization_damping_factor = 2.0; 
	/* Univariate Brent optimization tolerance factor (multiple of objective tolerance) */
	double brent_tolerance_factor = 1.;
	/* Left boundary of the Brent search interval. Set < 0 to enable reverse moves. */
	double brent_boundary_left = -1.0;
	/* Right boundary of the Brent search interval. */
	double brent_boundary_right = 2.0;
	/* Search policy for Brent interval restriction. If objective(brent_boundary_right)
	is not finite, the boundary is multiplied by this value. Same for the left boundary. 
	Larger values will find a feasible interval quicker but may restrict too aggressively. */
	double brent_feasible_search_restriction_factor = 0.75;
};

/* State information (optionnally) saved with each optimization iteration */
struct SolverState {
	size_t iter = 0;
	double objective_initial = -INFINITY;
	double objective_final = -INFINITY;
	/* Selected regularization */
	double lambda = 0.0;
	/* Starting parameters */
	std::vector< double > parameters;
	/* Gradient */
	std::vector< double > gradient;
	/* Hessian (dense) */
	std::vector< double > hessian;
	/* Selected direction vector */
	std::vector< double > direction;
	/* Left Brent boundary */
	double brent_left = 0.0;
	/* Right Brent boundary */
	double brent_right = 0.0;
	/* Selected optimal amplitude */
	double optstep = 0.0;
	/* No. of function evaluations */
	size_t n_eval = 0;
	/* No. of sparse solver calls */
	size_t n_solves = 0;
	/* No. of regularization attempts */
	size_t n_regul = 0;

	SolverState& print();
};

/* Implements optimization */
class Solver {

	friend struct BrentOptimizeObjectiveFunctor;

private:
	/* Stores unmodified parameter values, used during iterations and backtracking line search */
	std::vector< double > param_buffer;
	/* Stores hessian non-zero entries, used during regularization. */
	std::vector< double > hessian_buffer;
	/* Stores unmodified Newton direction, used during iterations and backtracking line search*/
	std::vector< double > direction_buffer;
	/* Vector of fixed parameters */
	std::vector< index_t > fixed_parameter_indices;
public:
	Trace& trace;
	SolverConfig config;
	/* Holds optional diagnostics after optimization, if config.diagnostic_mode == true.*/
	std::vector< SolverState > states;
	Solver(Trace& trace, SolverConfig config = {}) :
		trace{ trace }, config{ config },
		param_buffer(trace.tape.input_size()), direction_buffer(trace.tape.input_size())
	{};
	/* Solves the optimization problem */
	Solver& maximize();
	/* Set indices of fixed parameters */
	Solver& set_fixed_parameter_indices(const std::vector< index_t > indices) {
		assert(indices.size() < trace.tape.input_size()); // Don't fix all inputs...
		this->fixed_parameter_indices = indices;
		return *this;
	}
	/*Get indices of fixed parameters */
	std::vector< index_t > get_fixed_parameter_indices() const {
		return this->fixed_parameter_indices;
	}
	/* No. of forward tape evaluations */
	size_t n_eval_forward = 0;
	/* No. of reverse tape evaluations */
	size_t n_eval_reverse = 0;
};