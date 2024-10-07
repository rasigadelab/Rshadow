/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once

#include "tape.h"
#include "tensormap.h"

#include <stdexcept>

/******************************************************************
*** SPY CLASS *****************************************************
*******************************************************************/

/* Drop-in replacement for a 'double' variable or a tensor.

A Spy will record operations onto its associated tape, performing optimization
on-the-fly by selecting the most appropriate operator.

Mixes of spies and regular double's (literals
or variables) can appear in expressions. The regular double's will be recorded
as constants onto the tape.*/


class Spy : public Tensor {
private:
	/* Local index = first index in the associated tape */
	index_t tape_id_ = -1;
	/* Global index = index common to several tapes, typically a TensorMap index */
	index_t tensor_id_ = -1;

public:
	/* The tape onto which the detected operations will be recorded. */
	Tape& tape;

	/*** STANDARD API *************************************************************************************/

	/* These constructors should be called only by Spy-overloaded functions, hence not for declaring
	input spies. */

	/* Register a tensor */
	Spy(const Tensor& tensor, Tape& tape, const index_t tape_id) :
		Tensor{ tensor }, tape{ tape }, tape_id_{ tape_id }	{}
	/* Register a scalar */
	Spy(const double val, Tape& tape, const index_t tape_id) :
		Spy(Tensor{val}, tape, tape_id) {}
	/* Register a vector */
	Spy(const std::vector< double >& val, Tape& tape, const index_t tape_id) :
		Spy(Tensor{ val }, tape, tape_id) {}

	/*** AUTOMATIC INDEX FOR VARIABLE DECLARATION *****************************************************/
	
	/* Register a tensor, automatic index */
	Spy(const Tensor& tensor, Tape& tape) : 
		Spy(tensor, tape, index_t(tape.input_size()))
	{
		if (tape.operations.size() > 0) {
			throw std::logic_error("Attempt to declare an input Spy after recording started.");
		}
		/* Tape housekeeping */
		tape.n_input_size += val.size();
		tape.n_trace_size += val.size();
		tape.initial_values.reserve(tape.n_input_size);
		for (const double& x : val) {
			tape.initial_values.push_back(x);
		}
	}
	/* Register a scalar, automatic index */
	Spy(const double val, Tape& tape) :
		Spy(Tensor{ val }, tape) {}
	/* Register a vector, automatic index */
	Spy(const std::vector< double >& val, Tape& tape) :
		Spy(Tensor{ val }, tape) {}

	/*** TENSOR MAP API ***********************************************************************************/

	/* Register directly from a mapped tensor */
	Spy(const TensorMap& in, Tape& tape) : Tensor{ in.tensor() }, tape{tape}, tensor_id_{in.id()} {

		/* Check whether tensor is already registered in tape, as may arise after concatenation for instance */
		index_t existing_tape_id = tape.tape_id(tensor_id_);
		if (existing_tape_id > -1) {
			tape_id_ = existing_tape_id;
			return;
		}
		/* tensor is necessarily an input so the tape should be empty */
		assert(tape.operations.size() == 0 && "Attempt to declare an input after recording started.");
		tape_id_ = tape.n_input_size;
		tape.n_input_size += val.size();
		tape.n_trace_size += val.size();
		tape.map(tape_id_, tensor_id_);
		tape.initial_values.reserve(tape.n_input_size);
		for (const double& x : val) {
			tape.initial_values.push_back(x);
		}
	}

	/* Map an intermediate Spy value onto a mapped tensor. */
	void map(TensorMap& in, const bool write = true) {
		/* check that tensor is not mapped yet */
		assert(tape.tape_id(in.id()) < 0);
		tensor_id_ = in.id();
		tape.map(tape_id_, tensor_id_);
		if (write == true) {
			in = Tensor(*this);
		}
	}

	/*** COPY AND ELEMENT ACCESS  **************************************************************************************/

	/* Spy copy records an Identity operation. This is used
	to move indices around and define vectors. */
	Spy(const Spy& spy) : Spy(static_cast<const Tensor&>(spy), spy.tape, -1) {
		this->tape_id_ = this->is_scalar() ?
			tape.rec< IdentityScalar >(spy) :
			tape.rec< IdentityVector >(spy);
	}

	/* Element access should mimick exactly the behavior of the underlying Tensor class, but return
	a new Spy instead. Debug checks etc are performed when calling Tensor::vec_index() method,
	so the underlying Tensor subsetting should not be called again, we should reuse the provided
	vectorized index vi. */

	/* Bracket notation access */
	Spy operator[](const index_t i) const {
		assert(vec_index_is_valid(i));
		return Spy(val[i], tape, tape_begin() + i);		
	}
	/* Vector access */
	Spy operator()(const index_t i) const {
		return operator[](i);
	}
	/* Matrix access */
	Spy operator()(const index_t i, const index_t j) const {
		const index_t vi = vec_index(i, j);
		return operator[](vi);
	}
	/* 3D access */
	Spy operator()(const index_t i, const index_t j, const index_t k) const {
		const index_t vi = vec_index(i, j, k);
		return operator[](vi);
	}
	/* Arbitrary tensor access */
	Spy operator()(const std::vector< index_t >& ivec) const {
		const index_t vi = vec_index(ivec);
		return operator[](vi);
	}

	Tensor& tensor() {
		return static_cast<Tensor&>(*this);
	}
	const Tensor& tensor() const {
		return static_cast<const Tensor&>(*this);
	}

	/*** ACCESS AND CONVERSION TO TAPE/TRACE INDICES ********************************************************************/

	/* Getters */
	index_t tape_begin() const { return this->tape_id_; }
	index_t tape_end() const { return this->tape_id_ + this->val.size(); }
	index_t tensor_id() const { return this->tensor_id_; }

	operator IndexRange() const {
		return IndexRange(tape_begin(), tape_end());
	}
	operator OpIn::Range() const {
		return OpIn::Range{ tape_begin(), tape_end() };
	}
	operator OpIn::Scalar() const {
		assert(size() == 1);
		return OpIn::Scalar{ tape_begin() };
	}
	operator OpIn::ScalarScalar() const {
		assert(size() == 1);
		return OpIn::ScalarScalar{ tape_begin(), tape_begin() + 1 };
	}
	operator OpIn::InTensor<2>() const {
		return OpIn::InTensor<2>{ 
			IndexRange(tape_begin(), tape_end()),
			{ dim[0], dim[1] }
		};
	}

	/*** RECOMPILATION ***********************************************************************************/

	/* TODO This hould be moved out of Spy class. A tape recompilator may expose
	a dedicated vector. */

	/* Set elements to true in the freedom vector (free inputs) */
	void mark_as_free(std::vector< bool >& freedom_vector) const {
		assert(tape_end() <= index_t(freedom_vector.size()));
		for (index_t i = tape_begin(); i < tape_end(); i++) {
			freedom_vector[i] = true;
		}
	}
	/* Set elements to false in the freedom vector (constants) */
	void mark_as_fixed(std::vector< bool >& freedom_vector) const {
		assert(tape_end() <= index_t(freedom_vector.size()));
		for (index_t i = tape_begin(); i < tape_end(); i++) {
			freedom_vector[i] = false;
		}
	}

	/*** CHECK ARGUMENTS ***********************************************************************************/

	/* This is used by Spy overload functions */
private:
	/* Used for is_spy_arg_v constexpr test */
	/* Test whether a type in a tuple of types */
	template <typename T, typename Tuple>
	struct has_type;

	template <typename T>
	struct has_type<T, std::tuple<>> : std::false_type {};

	template <typename T, typename U, typename... Ts>
	struct has_type<T, std::tuple<U, Ts...>> : has_type<T, std::tuple<Ts...>> {};

	template <typename T, typename... Ts>
	struct has_type<T, std::tuple<T, Ts...>> : std::true_type {};

	template <typename T, typename Tuple>
	static inline constexpr bool tuple_contains_v = has_type<T, Tuple>::value;
public:
	/* Test whether T is an admissible argument to a spy function. Use in static_assert() or std::enable_if */
	template <typename T>
	static inline constexpr bool is_spy_arg_v = tuple_contains_v< T, std::tuple<
		double,
		Tensor,
		Spy
	> >;

	/*** LINEAR ALGEBRA ***********************************************************************************/

	Spy& make_col_vector() {
		static_cast<Tensor&>(*this).make_col_vector();
		return(*this);
	}
	Spy& make_row_vector() {
		static_cast<Tensor&>(*this).make_row_vector();
		return(*this);
	}
};