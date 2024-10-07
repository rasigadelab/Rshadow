/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "likelihood_methods.h"

#include "brent_optimize.h"
#include "rmath_bridge.h"
//#include "ASA/asa091.hpp" // Inverse CHI2 distribution

//#ifdef MATHLIB_STANDALONE
//double qchisq(double p, double df, int lower_tail, int log_p);
//#include "../vendor/rmath/Rmath.h"
//#endif

/*******************************************************************/
/*** STATIC CODE ***************************************************/
/*******************************************************************/

/* Obtain a pointer to the OpConst::Array<1> value of
 a suitable operator variant. Use with std::visit() */
struct OpVariantVisitorGetPointerToConstantScalar {
	double* ptr = nullptr;
	template< typename T >
	void operator()(T&& op) {
		using OpType = std::remove_reference_t< T >;
		if constexpr (
			std::is_base_of_v< OpConst::Array<1>, OpType> ||
			std::is_base_of_v< OpConst::Vector, OpType>
			) {
			this->ptr = &(op.constant[0]);
		}
		else {
			assert(false && "Called a pointer to constant on an operator without constant term.");
		}
	}
};

/* Obtain a pointer to the OpConst::Array<1> value of
 a suitable operator variant. */
template< typename V >
static double& op_access_constant_scalar(V& op_variant) {
	OpVariantVisitorGetPointerToConstantScalar visitor{ nullptr };
	std::visit(visitor, op_variant);
	assert(visitor.ptr != nullptr);
	return *visitor.ptr;
}

/* Functor for profile likelihood objective. Objective is the squared difference
between the current likelihood and a target likelihood. */
struct ProfileLikelihoodObjectiveFunctor {
	/* Solver with a recompiled tape/trace, taking the value of interest fixed,
	typically as a constant in an operator. */
	Solver& solver;
	/* Non-const reference to the constant representing the input to be profiled.
	Typically obtained using op_access_constant_scalar() helper function. */
	double& value_of_interest;
	/* The log-likelihood at the optimum */
	double maximum_likelihood;
	/* Confidence interval coverage, typically 0.95 */
	double coverage;
	/* Target likelihood = Maximum likelihood minus qchisq(coverage, 1), representing the
	likelihood cut point for a likelihood ratio test (LRT) at level alpha = 1 - coverage */
	double target = NAN;

	ProfileLikelihoodObjectiveFunctor(Solver& solver, double& value_of_interest, double maximum_likelihood, double coverage = 0.95) :
		solver{ solver }, value_of_interest{ value_of_interest },
		maximum_likelihood{ maximum_likelihood }, coverage{ coverage }
	{
		assert(coverage > 0. && coverage < 1.);
		const double LRT_CHISQ_CUTPOINT_95_HALF = 1.920729410347062016129;
		const double likelihood_delta = (coverage == 0.95) ?
			LRT_CHISQ_CUTPOINT_95_HALF : 0.5 * rmath_qchisq(coverage, 1., 1, 0);

		target = maximum_likelihood - likelihood_delta;
	}
	/* Compute the log-likelihood at input 'x' */
	double loglik(double x) {
		value_of_interest = x;
		solver.maximize();
		return solver.trace.values.back();
	}

	/* Minimization objective is the squared difference
	  between the current log-likelihood and the target */
	double operator()(double x) {
		double diff = loglik(x) - target;
		return diff * diff;
	}
};

/*******************************************************************/
/*** EXPOSED CODE **************************************************/
/*******************************************************************/




std::vector< LikelihoodMethods::confint_t > LikelihoodMethods::confidence_intervals_asymptotic(const double& coverage) const {

	std::vector< double > standard_deviations = this->asymptotic_standard_deviations();
	std::vector< confint_t > triplets(standard_deviations.size());

	for (size_t i = 0; i < triplets.size(); i++) {
		triplets[i] = {
			solver.trace[i],
			inverse_of_normal_cdf(0.5 * (1. - coverage), solver.trace[i], standard_deviations[i]),
			inverse_of_normal_cdf(1. - 0.5 * (1. - coverage), solver.trace[i], standard_deviations[i]),
			coverage
		};
	}
	return triplets;
}


double& LikelihoodMethods::reference_op_constant(index_t op_index) {
	return op_access_constant_scalar(solver.trace.tape.operations[op_index]);
}

LikelihoodMethods::profile_likelihood_output_t LikelihoodMethods::confidence_interval_profile_likelihood(
	double& value_of_interest,
	double point_estimate, double maximum_likelihood,
	double confint_halfwidth_guess,
	double coverage
) {

	ProfileLikelihoodObjectiveFunctor functor(solver, value_of_interest, maximum_likelihood, coverage);

	/* LOWER CONFIDENCE BOUND */

	/* Reset parameter value */
	value_of_interest = point_estimate;
	/* Bracket the value */
	double lower_width = confint_halfwidth_guess;
	while (functor.loglik(point_estimate - lower_width) > functor.target) {
		lower_width *= 2.;
	}
	brent_opt_output_t opt_result_lower = brent_optimize(functor, point_estimate - lower_width, point_estimate);

	profile_likelihood_opt_info_t lower_lsopt_info{
		point_estimate - lower_width,
		functor.loglik(point_estimate - lower_width),
		opt_result_lower.objective,
		opt_result_lower.n_eval
	};

	/* UPPER CONFIDENCE BOUND */

	/* Reset parameter value */
	value_of_interest = point_estimate;
	/* Bracket the value */
	double upper_width = confint_halfwidth_guess;
	while (functor.loglik(point_estimate + upper_width) > functor.target) {
		upper_width *= 2.;
	}
	brent_opt_output_t opt_result_upper = brent_optimize(functor, point_estimate, point_estimate + upper_width);

	profile_likelihood_opt_info_t upper_lsopt_info{
		point_estimate + upper_width,
		functor.loglik(point_estimate + upper_width),
		opt_result_upper.objective,
		opt_result_upper.n_eval
	};

	profile_likelihood_output_t output{
		confint_t{
			point_estimate,
			opt_result_lower.min,
			opt_result_upper.min,
			coverage
		},
		lower_lsopt_info,
		upper_lsopt_info
	};
	return output;
}

std::vector< LikelihoodMethods::profile_likelihood_output_t > LikelihoodMethods::confidence_intervals_profile(const double coverage) {

	/* Store vector of optimal input values */
	std::vector< double > optimal_inputs(solver.trace.tape.input_size());
	for (size_t i = 0; i < solver.trace.tape.input_size(); i++) {
		optimal_inputs[i] = solver.trace[i];
	}
	/* Keep track of maximum likelihood value */
	double maximum_likelihood = solver.trace.result();

	/* Get asymptotic confidence intervals, used as initial guess */
	auto asymptotic_confints = confidence_intervals_asymptotic(coverage);

	std::vector< LikelihoodMethods::profile_likelihood_output_t > results(solver.trace.tape.input_size());

	/*** ITERATE OVER INPUTS ***/
	for (index_t input_index = 0; input_index < index_t(solver.trace.tape.input_size()); input_index++) {

		solver.set_fixed_parameter_indices({ input_index });

		/* Reference to the fixed value */
		double& value_of_interest = solver.trace[input_index];

		double point_estimate = optimal_inputs[input_index];
		double confint_guess = 0.5 * (
			asymptotic_confints[input_index].upper_bound - 
			asymptotic_confints[input_index].lower_bound);

		/* Obtain profile confidence interval */
		auto confint = confidence_interval_profile_likelihood(
			value_of_interest,
			point_estimate, maximum_likelihood,
			confint_guess, coverage
		);

		results[input_index] = confint;

	} // end for fixed_index

	/* Restore solver state */
	for (size_t i = 0; i < solver.trace.tape.input_size(); i++) {
		solver.trace[i] = optimal_inputs[i];
	}
	solver.set_fixed_parameter_indices({});
	solver.trace.play();

	return results;
}

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include "sparse_matrix.h"
#include "eigen_sparse_matrix.h"

using sparse_mat_t = Eigen::SparseMatrix< double, Eigen::ColMajor, Eigen::Index >;

std::vector< double > LikelihoodMethods::asymptotic_standard_deviations() const {

	EigenSparseMat Eigen_hessian(solver.trace.hessian);
	const Eigen::Index N = Eigen_hessian.cols();

	/* Negate the matrix because Chol wants positive definite mat
	while Hessian is negative negative (*should be* negative definite) */
	Eigen_hessian.negate();

	/* Sparse Cholesky decomposition */
	Eigen::SimplicialLLT< sparse_mat_t, Eigen::Upper, Eigen::NaturalOrdering< Eigen::Index > > cholesky;
	cholesky.analyzePattern(Eigen_hessian);
	cholesky.factorize(Eigen_hessian);
	if (cholesky.info() != Eigen::Success) {
		throw "Bad Hessian";
	}

	/* We need an identity matrix and vector for the final computations */
	sparse_mat_t eye(N, N);
	eye.setIdentity();

	/* Copy the lower triangle. May be avoided? */
	sparse_mat_t cholL = cholesky.matrixL();

	/* Generic LU solver should be replaced with a specialized triangular solver here */
	Eigen::SparseLU< sparse_mat_t, Eigen::COLAMDOrdering<Eigen::Index> > LU_solver;
	LU_solver.analyzePattern(cholL);
	LU_solver.factorize(cholL);
	if (LU_solver.info() != Eigen::Success) {
		throw "Bad Hessian";
	}

	/* Inverse of the Cholesky lower triangle */
	sparse_mat_t cholinv = LU_solver.solve(eye);

	/* Square component-wise elements before taking column sums */
	for (Eigen::Index i = 0; i < cholinv.data().size(); i++) {
		cholinv.valuePtr()[i] *= cholinv.valuePtr()[i];
	}

	/* Sum of squares */
	Eigen::MatrixXd ones(1, N);
	ones.setOnes();

	/* Row vector. This uses a map over the return vector */
	std::vector< double > output(N);
	Eigen::Map< Eigen::MatrixXd > Eigen_output(output.data(), 1, N);
	Eigen_output = ones * cholinv;

	/* Take square roots to return SDs */
	for (size_t i = 0; i < output.size(); i++) {
		output[i] = sqrt(output[i]);
	}
	return output;
}