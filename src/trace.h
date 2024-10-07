/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "sparse_matrix.h"
#include "tape.h"

/*
The Tape and Trace objects are the higher-level classes of the
SHADOw-C lib.

The Trace holds numbers and the Tape describes actions on these
numbers.
*/

/****************************************************************************
*** FUNCTION TRACE
*****************************************************************************/
class Spy; // Forward declaration for readers

/* Trace of function, gradient and Hessian evaluation. */
class Trace {
public:
	/* Reference to Tape.
	REMARK. This is forward declaration, the .tape member can only
	be used inside trace.cpp implementation file, not the trace.h header file. */
	Tape& tape;
	/* Function values */
	std::vector< double > values;
	/* Adjoints (gradient in reverse sweep) */
	std::vector< double > adjoints;
	/* Sparse Hessian map*/
	SparseSymMat hessian;

	Trace(Tape& tape) : tape( tape ), values( tape.trace_size() ), 
		adjoints( tape.trace_size() ), hessian( tape.input_size() ) 
	{
		std::copy(tape.initial_values.cbegin(), tape.initial_values.cend(), values.begin());
	}

	/*******************************************************************/
	/*** GETTERS / SETTERS *********************************************/
	/*******************************************************************/

	/* Non-const access to a trace value by index */
	double& operator[](size_t i) {
		return values[i];
	}
	/* Const access to a trace value by index */
	double operator[](size_t i) const {
		return values[i];
	}
	/* Const access to the last value, aka the result */
	double result() const {
		return values.back();
	}
	/* First order partial derivative of the scalar trace result (gradient) */
	double partial(const index_t j) const {
		return adjoints[j];
	}
	/* Second order partial derivative of the scalar trace result (Hessian) */
	double partial(const index_t j, const index_t k) const {
		return hessian.read(j, k);
	}

	/*******************************************************************/
	/*** SPY READ ******************************************************/
	/*******************************************************************/
	double read_scalar(const Spy&) const;
	std::vector< double > read(const Spy&) const;
	Tensor read_tensor(const Spy&) const;

	/*******************************************************************/
	/*** COMPUTATION ***************************************************/
	/*******************************************************************/

	/* Compute function values in forward pass */
	Trace& play_forward();
	/* Compute gradient and Hessian values in reverse pass */
	Trace& play_reverse();
	/* Compute function values, gradient, and Hessian */
	Trace& play();
};