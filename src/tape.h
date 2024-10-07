/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once


#include "operators.h"
#include "tensormap.h"

class Trace;

#include <variant>
#include <optional>
#include <unordered_map>

/* 
The Tape and Trace objects are the higher-level classes of the
SHADOw-C lib.

The Trace holds numbers and the Tape describes actions on these
numbers.
*/

/****************************************************************************
*** TAPE (EXPRESSION GRAPH)
*****************************************************************************/

/* Holds a sequence of operations */
class Tape {

	friend class Spy;
protected:
	/* Size of the input vector (sum of the sizes of input tensors) */
	size_t n_input_size = 0;
	/* Total size of the trace vector (size of input vector + sum of the sizes of all operator outputs) */
	size_t n_trace_size = 0;

private:
	/* Convert a tuple of types to a variant */
	template< typename T > struct as_variant;
	/* Convert a tuple of types to a variant */
	template< typename... U> struct as_variant< std::tuple< U... > > {
		using type = std::variant< U... >;
	};

	/* https://stackoverflow.com/questions/45892170/how-do-i-check-if-an-stdvariant-can-hold-a-certain-type */

	/* Test whether a variant holds type T as an alternative */
	template<typename T, typename VARIANT_T>
	struct isVariantMember;
	/* Test whether a variant holds type T as an alternative */
	template<typename T, typename... ALL_T>
	struct isVariantMember<T, std::variant<ALL_T...>>
		: public std::disjunction<std::is_same<T, ALL_T>...> {};

public:
	Tape(const size_t n_input_size = 0) : n_input_size{ n_input_size }, n_trace_size{ n_input_size }
	{
		initial_values.reserve(n_input_size);
	}

	using OpVariant = as_variant< operator_types >::type;
	std::vector< OpVariant > operations;

	/* Size of the input vector (sum of the sizes of input tensors) */
	size_t input_size() const {
		return n_input_size;
	}
	/* Total size of the trace vector (size of input vector + sum of the sizes of all operator outputs) */
	size_t trace_size() const {
		return n_trace_size;
	}

	/****************************************************************************
	*** RECORDING THE TAPE
	*****************************************************************************/

	/* Append to tape using known output indices */
	template< typename OP >
	void op_register(const OP operation) {
		static_assert(isVariantMember< OP, OpVariant >::value, "Unknown operator type, did you forget to declare it?");
		operations.push_back(operation);
		n_trace_size += operation.out.size();
	}

	/* Record an operation (append to tape) and return the index (indices) of its output. 
	Alternative version with explicit input/output types (not generic types but strongly typed
	in/out) */
	template< typename OP >
	index_t rec(const typename OP::OpInType in, const typename OP::OpConstantType constant = {}) {

		index_t out_begin = index_t(trace_size());

		if constexpr (std::is_base_of_v< OpOut::Scalar, OP >) {
			OpOut::Scalar out({ out_begin });
			OP operation(in, constant, out);
			op_register(operation);
		}
		if constexpr (std::is_base_of_v< OpOut::Range, OP >) {
			size_t out_size = OP::output_range_size(in, constant);
			OpOut::Range out(out_begin, out_begin + out_size);
			OP operation(in, constant, out);
			op_register(operation);
		}
		return out_begin;
	}

#ifdef DEPREC
	/* QoL overloads */
	template< typename OP >
	index_t rec(const typename OP::OpInType in, const Tensor constant) {
		return rec< OP >(in, typename OP::OpConstantType(constant));
	}
#endif
	/****************************************************************************
	*** PLAYING THE TAPE
	*****************************************************************************/

	/* MARKED FOR DEPRECATION, should now move to Trace object */

	/* Compute function values in forward pass */
	void play_forward(Trace& trace) const;

	/* Compute gradient and Hessian values in reverse pass */
	void play_reverse(Trace& trace) const;

	/* Compute all */
	void play(Trace& trace) const;



	/****************************************************************************
	*** MANIPULATING THE TAPE - RECOMPILATION, CONCATENATION
	*****************************************************************************/

	/* Holds values of Spy inputs declared for the tape */
	std::vector< double > initial_values;

#ifdef DEPREC
	/* Optional reindexing vector for recompiled tapes. Use reindex_from_master[ original_index ] to
	find the new index using the original index of an input in the master tape. The vector is empty
	if the tape has not been recompiled. The return value is -1 if the provided index is not
	a free variable. */
	std::vector< index_t > reindex_from_master;

	/* Optional reindexing vector for recomiled tapes. Use 
	operator_index_from_master[ original_index ], where original
	index is the index of a fixed input, to find the index of
	the operator holding the original values as a constant. */
	std::vector< index_t > operator_index_from_master;

	/* Optional reindexing vector for concatenated tapes. Use 
	reindex_from_parts[ part_tape_id ][ var_id_in_part ] to find the new using
	the index of the original part tape and the index of the variable within
	this original part tape. */
	std::vector< std::vector< index_t > > reindex_from_parts;
	
	/* Return a boolean vector of input freedom indicators, to be passed
	to the Tape::recompile() method after fixing some input. */
	std::vector< bool > get_input_freedom_vector() const {
		std::vector< bool > freedom_vector(input_size());
		for (size_t i = 0; i < input_size(); i++) freedom_vector[i] = true;
		return freedom_vector;
	}

	/* Returns true if the tape has not been recompiled */
	bool is_master_tape() const {
		return reindex_from_master.size() == 0;
	}

	/* Recompile as a new Tape after fixing parameters */
	Tape recompile(const std::vector< bool >& input_freedom_states, const std::vector< double >& input_values) const;

	Tape recompile(const std::vector< bool >& input_freedom_states) const {
		return this->recompile(input_freedom_states, this->initial_values);
	}

	Tape recompile(const std::vector< bool >& input_freedom_states, const TensorMap& map) const {
		/* Fetch the inputs directly from the TensorMap */
		std::vector< double > inputs(input_freedom_states.size());
		for (const auto& it : to_tensor_map) {
			if (it.first >= index_t(inputs.size())) continue;
			inputs[it.first] = map[it.second].tensor().scalar();
		}
		return this->recompile(input_freedom_states, inputs);
	}

	/* Concatenate a list of Tapes into a new Tape */
	static Tape concatenate(std::vector< Tape* > tape_list);
#endif
	/**********************************************************/
	/*** TENSOR MAPPING ***************************************/
	/**********************************************************/
private:
	/* Tape to tensor */
	std::unordered_map< index_t, index_t > to_tensor_map;
	/* Tensor to tape */
	std::unordered_map< index_t, index_t > to_tape_map;

public:
	void map(const index_t tape_id, const index_t tensor_id) {
		to_tensor_map.insert_or_assign(tape_id, tensor_id);
		to_tape_map.insert_or_assign(tensor_id, tape_id);
	}

	index_t tensor_id(const index_t tape_id) const {
		const auto it = to_tensor_map.find(tape_id);
		return it == to_tensor_map.end() ? -1 : it->second;
	}

	index_t tape_id(const index_t tensor_id) const {
		const auto it = to_tape_map.find(tensor_id);
		return it == to_tape_map.end() ? -1 : it->second;
	}

	void write_tensor_map_to_trace(Trace& trace, const TensorMap& map) const;
	void write_trace_to_tensor_map(const Trace& trace, TensorMap& map) const;

	Trace get_trace(const TensorMap& map);
};