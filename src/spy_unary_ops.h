/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "spy.h"

/*** UNARY NEGATION **********************************************/
Tensor operator-(const Tensor&);
Spy    operator-(const Spy&);
/*** NATURAL LOG *************************************************/
Tensor log(const Tensor&);
Spy    log(const Spy&);
/*** LOG(1 + X) **************************************************/
Tensor log1p(const Tensor&);
Spy    log1p(const Spy&);
/*** LOG(1 - X) **************************************************/
Tensor log1m(const Tensor&);
Spy    log1m(const Spy&);
/*** EXP *********************************************************/
Tensor exp(const Tensor&);
Spy    exp(const Spy&);
/*** LOG GAMMA****************************************************/
Tensor lgamma(const Tensor&);
Spy    lgamma(const Spy&);
/*** LOGIT *******************************************************/
Tensor logit(const Tensor&);
Spy    logit(const Spy&);
/*** LOGISTIC ****************************************************/
Tensor logistic(const Tensor&);
Spy    logistic(const Spy&);
/*** SIN *********************************************************/
Tensor sin(const Tensor&);
Spy    sin(const Spy&);
/*** COS *********************************************************/
Tensor cos(const Tensor&);
Spy    cos(const Spy&);