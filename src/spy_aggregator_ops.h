/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "spy.h"

/*** AGGREGATOR OPS *******************************************/

/* Sum of a vector, matrix or tensor */
Tensor sum(const Tensor& a);
Spy    sum(const Spy& a);

/* Sum of squares of a vector, matrix or tensor */
Tensor sumsq(const Tensor& a);
Spy    sumsq(const Spy& a);

/* Dot product of two vectors */
Tensor dot(const Tensor& a, const Tensor& b);
Spy    dot(const Spy&    a, const Spy&    b);
Spy    dot(const Spy&    a, const Tensor& b);
Spy    dot(const Tensor& a, const Spy&    b);

/* Sum of Bernoulli log-likelihoods. b must be binary. */
Tensor sum_log_dbern(const Tensor& a, const Tensor& b);
Spy    sum_log_dbern(const Spy& a, const Tensor& b);