/***********************************************************************************/
/*  ________  ___  ___  ________  ________  ________  ___       __                 */
/* |\   ____\|\  \|\  \|\   __  \|\   ___ \|\   __  \|\  \     |\  \               */
/* \ \  \___|\ \  \\\  \ \  \|\  \ \  \_|\ \ \  \|\  \ \  \    \ \  \              */
/*  \ \_____  \ \   __  \ \   __  \ \  \ \\ \ \  \\\  \ \  \  __\ \  \             */
/*   \|____|\  \ \  \ \  \ \  \ \  \ \  \_\\ \ \  \\\  \ \  \|\__\_\  \            */
/*     ____\_\  \ \__\ \__\ \__\ \__\ \_______\ \_______\ \____________\           */
/*    |\_________\|__|\|__|\|__|\|__|\|_______|\|_______|\|____________|           */
/*    \|_________|                                                                 */
/*                                                                                 */
/* >>>>> A Sparse Hessian Automatic Differentiation Optimization floW <<<<<<<<<<<< */

/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once

#include "op_base.h"
#include "op_aggregate.h"
#include "op_unary.h"
#include "op_plus.h"
#include "op_minus.h"
#include "op_multiply.h"
#include "op_divide.h"
#include "op_power.h"
#include "op_iverson.h"

/****************************************************************************
*** OPERATOR DECLARATION
*****************************************************************************/

//https://stackoverflow.com/questions/53394100/concatenating-tuples-as-types

/* Concatenate tuples */
template<typename ... input_t>
using tuple_cat_t = decltype(std::tuple_cat( std::declval<input_t>()... ));

using operator_types = tuple_cat_t<
	op_addition_types, op_minus_types, op_multiply_types, op_divide_types, op_power_types,
	op_aggregate_types, op_unary_types, op_iverson_types
>;