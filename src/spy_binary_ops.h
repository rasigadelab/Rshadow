/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once
#include "spy.h"

/******************************************************************
*** OPERATOR/FUNCTION OVERLOADS ***********************************
*******************************************************************/

/* Arithmetic type overloads for types that are implicitly convertible to double */
#define SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD(FUNC) \
template< typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type > \
static Spy FUNC(const T& a, const Spy& b) { return FUNC(double(a), b); } \
template< typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type > \
static Spy FUNC(const Spy& a, const T& b) { return FUNC(a, double(b)); }

/*** ADDITION *****************************************************/
Tensor operator+(const Tensor& a, const Tensor& b);
Spy operator+(const Spy& a, const Spy& b);
Spy operator+(const Spy& a, const Tensor& b);
Spy operator+(const Spy& a, const double& b);
Spy operator+(const Tensor& a, const Spy& b);
Spy operator+(const double& a, const Spy& b);
SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD(operator+);

/*** SUBTRACTION *************************************************/
Tensor operator-(const Tensor& a, const Tensor& b);
Spy operator-(const Spy& a, const Spy& b);
Spy operator-(const Spy& a, const Tensor& b);
Spy operator-(const Spy& a, const double& b);
Spy operator-(const Tensor& a, const Spy& b);
Spy operator-(const double& a, const Spy& b);
SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD(operator-);

/*** MULTIPLICATION ***********************************************/
Tensor operator*(const Tensor& a, const Tensor& b);
Spy operator*(const Spy& a, const Spy& b);
Spy operator*(const Spy& a, const Tensor& b);
Spy operator*(const Spy& a, const double& b);
Spy operator*(const Tensor& a, const Spy& b);
Spy operator*(const double& a, const Spy& b);
SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD(operator*);

/*** DIVISION ****************************************************/
Tensor operator/(const Tensor& a, const Tensor& b);
Spy operator/(const Spy& a, const Spy& b);
Spy operator/(const Spy& a, const Tensor& b);
Spy operator/(const Spy& a, const double& b);
Spy operator/(const Tensor& a, const Spy& b);
Spy operator/(const double& a, const Spy& b);
SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD(operator/);

/*** POWER ****************************************************/
Tensor pow(const Tensor& a, const Tensor& b);
Spy pow(const Spy& a, const Spy& b);
Spy pow(const Spy& a, const Tensor& b);
Spy pow(const Spy& a, const double& b);
Spy pow(const Tensor& a, const Spy& b);
Spy pow(const double& a, const Spy& b);
SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD(pow);

/*** GREATER THAN**********************************************/
Tensor operator>(const Tensor& a, const Tensor& b);
Spy operator>(const Spy& a, const Spy& b);
Spy operator>(const Spy& a, const Tensor& b);
Spy operator>(const Spy& a, const double& b);
Spy operator>(const Tensor& a, const Spy& b);
Spy operator>(const double& a, const Spy& b);
SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD(operator>);

/*** GREATER THAN OR EQUAL *************************************/
Tensor operator>=(const Tensor& a, const Tensor& b);
Spy operator>=(const Spy& a, const Spy& b);
Spy operator>=(const Spy& a, const Tensor& b);
Spy operator>=(const Spy& a, const double& b);
Spy operator>=(const Tensor& a, const Spy& b);
Spy operator>=(const double& a, const Spy& b);
SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD(operator>=);

/*** LESS THAN**********************************************/
Tensor operator<(const Tensor& a, const Tensor& b);
Spy operator<(const Spy& a, const Spy& b);
Spy operator<(const Spy& a, const Tensor& b);
Spy operator<(const Spy& a, const double& b);
Spy operator<(const Tensor& a, const Spy& b);
Spy operator<(const double& a, const Spy& b);
SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD(operator<);

/*** LESS THAN OR EQUAL *************************************/
Tensor operator<=(const Tensor& a, const Tensor& b);
Spy operator<=(const Spy& a, const Spy& b);
Spy operator<=(const Spy& a, const Tensor& b);
Spy operator<=(const Spy& a, const double& b);
Spy operator<=(const Tensor& a, const Spy& b);
Spy operator<=(const double& a, const Spy& b);
SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD(operator<=);

#undef SHADOW_SPY_ARITHMETIC_TYPE_OVERLOAD