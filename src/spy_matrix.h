/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "spy.h"

Tensor matmult(const Tensor& a, const Tensor& b);
Spy matmult(const Spy& a, const Spy& b);
Spy matmult(const Spy& a, const Tensor& b);
Spy matmult(const Tensor& a, const Spy& b);