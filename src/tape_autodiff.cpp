/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#include "tape.h"
#include "trace.h"

/*
For mathematical details of Hessian automatic differentiation please refer
to https://link.springer.com/article/10.1007/s12532-016-0100-3

Mu Wang, Assefaw Gebremedhin & Alex Pothen. Capitalizing on live variables:
new algorithms for efficient Hessian computation via automatic differentiation.
Math. Prog. Comp. (2016) 8:393–433. DOI 10.1007/s12532-016-0100-3

The automatic differentiation algorithm follows the general idea of the so - called 
Basic Reverse Mode Hessian algorithm described in Figure 2, with modifications to account
for our particular use case. We make a distinction between nodes pairs that are both operands
of the current node(3rd case of Expression 7); pairs in which exactly one node is an operand
(2nd case of Expression 7) and we completely skip non - operand pairs in the live set
(1st case of Expression 7). This is similar in spirit with the so - called enhanced version of
the algorithm(Figure 4) but with less systematic checks and optimizations to favor code
readability and maintenance over brute performance.

Gradient is computed as a standard adjoint mode.
*/

/* Overload pattern for variant identification */

template<typename ... Ts>                                                 
struct Overload : Ts ... {
	using Ts::operator() ...;
};
template<class... Ts> Overload(Ts...) -> Overload<Ts...>;

/****************************************************************************
*** FORWARD SWEEP
*****************************************************************************/

/* The forward sweep of the CpGraph computes function values and stores them
in the first slot of the dual numbers. */

/* HINT: additional visitor arguments are 'part of the visitor' so they're
given to the constructor, not the operator. */

/* foward OpVariant visitor */
struct OpVariantForward {
	std::vector< double >& values;
	OpVariantForward(std::vector< double >& values) : values{ values } {}
	template< typename T > void operator()(T&& op) {
		op.evaluate(values);
	}
};

/* Compute function values in forward pass */
void Tape::play_forward(Trace& trace) const {

	for (const auto& op_variant : operations) {
		std::visit(OpVariantForward(trace.values), op_variant);
	}
}

/****************************************************************************
*** REVERSE SWEEP
*****************************************************************************/

/* So-called creating part, unary case,
h(j,j) += d²i/dj².a(i)     
*/
template< typename OpType >
static void creating_part_unary(
	OpType& op, typename OpType::LocalDiff& local, Trace& trace, const double w,
	const index_t i_local, const index_t j_local
) {
	if constexpr (std::is_base_of_v< HessianDiagAlwaysZero, OpType > == false) {
		const index_t j = op.in[j_local];
		const double d2idj2 = local.partial(i_local, j_local, j_local);
		if (d2idj2 != 0.0) {
			trace.hessian.get(j, j) += d2idj2 * w;
		}
	}
}

/* So-called creating part, binary case,
h(j,j) += d²i/dj².a(i)  
h(k,k) += d²i/dk².a(i)
h(j,k) += d²i/djdk.a(i)
*/
template< typename OpType >
static void creating_part_binary(
	OpType& op, typename OpType::LocalDiff& local, Trace& trace, const double w, 
	const index_t i_local, const index_t j_local, const index_t k_local
) {
	const index_t j = op.in[j_local];
	const index_t k = op.in[k_local];

	if constexpr (std::is_base_of_v< HessianDiagAlwaysZero, OpType > == false) {
		const double d2idj2 = local.partial(i_local, j_local, j_local);
		if (d2idj2 != 0.0) {
			trace.hessian.get(j, j) += d2idj2 * w;
		}
		const double d2idk2 = local.partial(i_local, k_local, k_local);
		if (d2idk2 != 0.0) {
			trace.hessian.get(k, k) += d2idk2 * w;
		}
	}
	if constexpr (std::is_base_of_v< HessianOffDiagAlwaysZero, OpType > == false) {
		const double d2idjdk = local.partial(i_local, j_local, k_local);
		if (d2idjdk != 0.0) {
			trace.hessian.get(j, k) += d2idjdk * w;
		}
	}
}

/* Reverse OpVariant visitor */
struct OpVariantReverse {

	Trace& trace;

	OpVariantReverse(Trace& trace) : trace{ trace } {}

	template< typename T >
	void operator()(T&& op) {

		/* The forwarding reference in argument hampers some static tests so
		we need to remove a ref */
		using OpType = std::remove_reference_t< T >; 

		/* Index of the output node. Local derivatives are those of the
		output values wrt to the input(s) j, k, etc defined later. */
		
		for (index_t i_local = 0; i_local < index_t(op.out.size()); i_local++) {
			const index_t i = op.out[i_local];

			/* All gradients and Hessian elements are obtained by calls to
			OpType::LocalDiff.partial(i, j) or .partial(i, j, k). In general,
			operators can harbor any combinations of length for the output
			(i-indexed) and the input (j,k-indexed).
			We use optimized code paths when peculiarities regarding partials
			are known at compile time. These peculiarities are signaled by the
			operator using flags (empty structs) such as "HessianDiagAlwaysZero".
			There are two types of peculiarities, "dimension" and "sparsity" ones.

			Dimension peculiarities signal that either the input or output is scalar
			for instance.

			Sparsity peculiarities signal that some combinations of in/out indices
			have always zero partials in the Jacobian or Hessian.
			*/

			/* Local gradient and Hessian */		
			typename OpType::LocalDiff local = op.local_diff(trace.values);

			/* Adjoint of i */
			const double w = trace.adjoints[i];

			/* ADJOINT UPDATE:
			
			a(j) += di/dj * a(j)
			
			=> here we let w = a(j) before the update.
			The step is skipped	if w = 0 or if di/dj = 0

			*/
			if (w != 0.0) {

				/* Unary input */
				if constexpr (std::is_base_of_v< OpIn::Array<1>, OpType >) {
					trace.adjoints[op.in[0]] += local.partial(i_local, 0) * w;
				}
				/* Binary input j,k */
				else if constexpr (std::is_base_of_v< OpIn::ScalarScalar, OpType >) {
					trace.adjoints[op.in.left[0] ] += local.partial(i_local, 0) * w;
					trace.adjoints[op.in.right[0]] += local.partial(i_local, 1) * w;
				}
				/* Element-wise input, only consider associated inputs */
				else if constexpr (std::is_base_of_v< OpIsElementWise, OpType>) {

					/* Single range, j = i */
					if constexpr (std::is_base_of_v< OpIn::Range, OpType >) {
						trace.adjoints[op.in[i_local]] += local.partial(i_local, i_local) * w;
					}
					/* Paired range, j = i, k = i+n */
					else if constexpr (std::is_base_of_v< OpIn::RangePair, OpType >) {
						const size_t n = op.in.left.size();
						const index_t j_local = i_local;
						const index_t k_local = i_local + n;
						trace.adjoints[op.in[j_local]] += local.partial(i_local, j_local) * w; // j = i
						trace.adjoints[op.in[k_local]] += local.partial(i_local, k_local) * w; // j = i + n
					}
					/* Vector/scalar pair, j=i for vector, k=n for scalar */
					else if constexpr (std::is_base_of_v< OpIn::RangeArrayPair<1>, OpType >) {
						const size_t n = op.in.left.size();
						const index_t j_local = i_local;
						const index_t k_local = n;
						trace.adjoints[op.in[j_local]] += local.partial(i_local, j_local) * w; // j = i
						trace.adjoints[op.in[k_local]] += local.partial(i_local, k_local) * w; // j = n
					}
					/* Scalar/vector pair, j=0 for scalar, k=i+1 for vector */
					else if constexpr (std::is_base_of_v< OpIn::ArrayRangePair<1>, OpType >) {
						const size_t n = 1;
						const index_t j_local = 0;
						const index_t k_local = i_local + n;
						trace.adjoints[op.in[j_local]] += local.partial(i_local, j_local) * w; // j = i
						trace.adjoints[op.in[k_local]] += local.partial(i_local, k_local) * w; // j = i + n
					}
					else {
						assert(false && "Missing code path in element wise operator.");
					}
				}				
				/* Generic iterable input */
				else {
					for (size_t j_local = 0; j_local < op.in.size(); j_local++) {
						const double didj = local.partial(i_local, j_local);
						if (didj == 0.0) continue;
						const index_t j = op.in[j_local];
						trace.adjoints[j] += didj * w;
					}
				}
			}
				
			/* HESSIAN UPDATE:
			
			h(j,k) +=
				di/dk.h(i,j) + di/dj.h(i,k) + // Pushing part 1
				di/dj.di/dk.h(i,i) +          // Pushing part 2
				d²i/djdk.a(i)                 // Creating part

			Following terminology of Wang 2016, by reference to the previous
			"edge pushing" algorithm, the first line is called "pushing part 1",
			the second line, "pushing part 2", the last line, "creating part".

			Each part is skipped whenever a term is zero.
			
			*/
			auto r = trace.hessian.get_row_ptr(i);

			/* PUSHING PART 1 - r(v_j) != 0 implies h(v_j,v_i) != 0 and j != i*/
			if (r != nullptr) for (const auto& it_j : *r) {
				/* j is a variable in the active set, j <> i */
				const index_t j = it_j.first;
				if (j == i) continue;		

				/* Unary input */
				if constexpr (std::is_base_of_v< OpIn::Array<1>, OpType >) {				
					const double didk = local.partial(i_local, 0);
					if (didk != 0.0) {
						const index_t k = op.in[0];
						trace.hessian.get(j, k) += didk * it_j.second;
					}				
				}
				/* Binary input j,k - unrolled. First k may be pushed to the Unary block above */
				else if constexpr (std::is_base_of_v< OpIn::ScalarScalar, OpType >) {
					const double didk = local.partial(i_local, 0);
					if (didk != 0.0) {
						const index_t k = op.in.left[0];
						trace.hessian.get(j, k) += didk * it_j.second;
					}
					const double didl = local.partial(i_local, 1);
					if (didl != 0.0) {
						const index_t l = op.in.right[0];
						trace.hessian.get(j, l) += didl * it_j.second;
					}
				}
				/* Generic iterable input */
				else {					
					for (size_t k_in = 0; k_in < op.in.size(); k_in++) {
						/* k is an operand, k != j, di/dk <> 0 */
						const index_t k = op.in[k_in];
						const double didk = local.partial(i_local, k_in);
						if (didk == 0.0) continue;
						trace.hessian.get(j, k) += didk * it_j.second;
					}
				}
			}

			/* PUSHING PART 2 - r(v_i) != 0 */
			if (r != nullptr) {
				auto it_i = r->find(i);
				if (it_i != r->end()) {

					const double r_i = it_i->second;
					assert(r_i != 0.0); // Should not happen, but anyway...

					/* Unary input */
					if constexpr (std::is_base_of_v< OpIn::Array<1>, OpType >) {
						const double didj = local.partial(i_local, 0);
						if (didj != 0.0) {
							const index_t j = op.in[0];
							trace.hessian.get(j, j) += didj * didj * r_i;
						}
					}
					/* Binary input j,k - unrolled. First k may be pushed to the Unary block above */
					else if constexpr (std::is_base_of_v< OpIn::ScalarScalar, OpType >) {
						const double didj = local.partial(i_local, 0);
						const double didk = local.partial(i_local, 1);
						const double didj_didk = didj * didk;
						if (didj != 0.0) {
							const index_t j = op.in.left[0];
							trace.hessian.get(j, j) += didj * didj * r_i;
						}
						if (didk != 0.0) {
							const index_t k = op.in.right[0];
							trace.hessian.get(k, k) += didk * didk * r_i;
						}
						if (didj_didk != 0.0) {
							const index_t j = op.in.left[0];
							const index_t k = op.in.right[0];
							trace.hessian.get(j, k) += didj_didk * r_i;
						}
					}
					/* Nary input */
					else {
						/* Unordered pairs of operands (j,k), including j == k */
						for (size_t j_in = 0; j_in < op.in.size(); j_in++) {

							const double didj = local.partial(i_local, j_in);
							if (didj == 0.0) continue;

							for (size_t k_in = j_in; k_in < op.in.size(); k_in++) {

								const double didk = local.partial(i_local, k_in);
								if (didk == 0.0) continue;

								const index_t j = op.in[j_in];
								const index_t k = op.in[k_in];

								trace.hessian.get(j, k) += didj * didk * r_i;
							} // END for k_in
						} // END for j_in
					}
				} // END if it_i != r->end()
			} // END if r != nullptr

			/* CREATING PART */
			if (w != 0.0) {
				/* Unary input */
				if constexpr (std::is_base_of_v< OpIn::Array<1>, OpType >) {				
					creating_part_unary(op, local, trace, w, i_local, 0);
				}
				/* Binary input j,k */
				else if constexpr (std::is_base_of_v< OpIn::ScalarScalar, OpType >) {
					creating_part_binary(op, local, trace, w, i_local, 0, 1);
				}
				/* Element-wise input, either Vector, VectorScalar, ScalarVector, VectorVector */
				else if constexpr (std::is_base_of_v< OpIsElementWise, OpType>) {

					/* Single range, j = i */
					if constexpr (std::is_base_of_v< OpIn::Range, OpType >) {
						creating_part_unary(op, local, trace, w, i_local, i_local);
					}
					/* Paired range, j = i, k = i+n */
					else if constexpr (std::is_base_of_v< OpIn::RangePair, OpType >) {
						const size_t n = op.in.left.size();
						creating_part_binary(op, local, trace, w, i_local, i_local, i_local + n);
					}
					/* Vector/scalar pair, j=i for vector, k=n for scalar */
					else if constexpr (std::is_base_of_v< OpIn::RangeArrayPair<1>, OpType >) {
						const size_t n = op.in.left.size();
						creating_part_binary(op, local, trace, w, i_local, i_local, n);
					}
					/* Scalar/vector pair, j=0 for scalar, k=i+1 for vector */
					else if constexpr (std::is_base_of_v< OpIn::ArrayRangePair<1>, OpType >) {
						creating_part_binary(op, local, trace, w, i_local, 0, i_local + 1);
					}
					else {
						assert(false && "Missing code path in element wise operator.");
					}
				}

				/* Generic N-ary input /!\ QUADRATIC TIME CODE PATH */
				else {
					for (size_t j_local = 0; j_local < op.in.size(); j_local++) {

						const index_t j = op.in[j_local];

						/* Local Hessian diagonal, skip when certainly zero */
						if constexpr (std::is_base_of_v< HessianDiagAlwaysZero, OpType > == false) {
							const double d2idj2 = local.partial(i_local, j_local, j_local);
							if (d2idj2 != 0.0) {
								trace.hessian.get(j, j) += d2idj2 * w;
							}
						} // END skip if HessianDiagAlwaysZero

						/* Local Hessian off-diagonal, skip when certainly zero */
						if constexpr (std::is_base_of_v< HessianOffDiagAlwaysZero, OpType > == false) {
							for (size_t k_local = j_local + 1; k_local < op.in.size(); k_local++) {
								const double d2idjdk = local.partial(i_local, j_local, k_local);
								if (d2idjdk != 0.0) {
									const index_t k = op.in[k_local];
									trace.hessian.get(j, k) += d2idjdk * w;
								}
							} // END for k_local (operand)
						} // END skip if HessianOffDiagAlwaysZero
					} // END for j_in (operand)
				}
			}
		
			/* HOUSEKEEPING PART: remove the ith row/column */
			if (r != nullptr) {
				trace.hessian.erase(i);
			}

		} // FOR i IN OUTPUTS

	} // end operator() body
};


/* Compute gradient and Hessian values in reverse pass */
void Tape::play_reverse(Trace& trace) const {

	/* Clear the gradient */
	for (auto& it : trace.adjoints) it = 0.0;

	/* Seed the gradient */
	trace.adjoints.back() = 1.0;

	/* Clear the Hessian */
	trace.hessian.matrix.clear();

	/* Reverse sweep of the computational graph */
	for (auto it = operations.crbegin(); it != operations.crend(); it++) {

		/* Update the gradient and Hessian */
		std::visit(OpVariantReverse(trace), *it);
	}
}

void Tape::play(Trace& trace) const {
	assert(trace.values.size() == trace_size());
	play_forward(trace);
	play_reverse(trace);
}