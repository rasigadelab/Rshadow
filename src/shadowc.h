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

#include "operators.h"
#include "tape.h"
#include "solver.h"
#include "tensor.h"
#include "spy_overloads.h"
#include "likelihood_methods.h"