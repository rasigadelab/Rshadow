/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "solver.h"

/* Implements a set of likelihood statistics-based methods including
asymptotic confidence intervals and profile likelihood confidence intervals. */
class LikelihoodMethods {
private:
	double& reference_op_constant(index_t op_index);

public:
	Solver& solver;
	LikelihoodMethods(Solver& solver) : solver{ solver } {}

	/* Output result for a coefficient interval */
	struct confint_t {
		double estimate = 0.;
		double lower_bound = 0.;
		double upper_bound = 0.;
		double coverage = 0.;
	};

	/*** ASYMPTOTIC NORMALITY METHOD ***/

	/* Return a vector of asymptotic Wald standard deviations of parameter
	estimation uncertainty in a maximum likelihood context. Use +/-1.96 SD
	to approximate confidence intervals. */
	std::vector< double > asymptotic_standard_deviations() const;

	/* Obtain a series of approximate confidence intervals based on the
	assumption of asymptotic normality of the parameters. */
	std::vector< confint_t > confidence_intervals_asymptotic(const double& coverage = 0.95) const;

	/*** PROFILE LIKELIHOOD METHOD ***/

	/* Diagnostic information on the results of Brent optimization */
	struct profile_likelihood_opt_info_t {
		double initial_bracket = 0.0;
		double initial_bracket_log_likelihood = 0.0;
		double residual_squared_diff = 0.0;
		size_t n_evaluations = 0;
	};

	/* Describe confidence interval as well as diagnostic values
		regarding the optimization procedure. */
	struct profile_likelihood_output_t : confint_t {
		/* Profile-specific results */
		profile_likelihood_opt_info_t lower;
		profile_likelihood_opt_info_t upper;
	};

private:
	/* Obtain the profile likelihood confidence interval for the constant
	parameter referenced in the 'value of interest' argument. */
	profile_likelihood_output_t confidence_interval_profile_likelihood(
		double& value_of_interest,
		double point_estimate, double maximum_likelihood,
		double confint_halfwidth_guess,
		double coverage = 0.95
	);
public:
	std::vector< profile_likelihood_output_t > confidence_intervals_profile(const double coverage = 0.95);
};