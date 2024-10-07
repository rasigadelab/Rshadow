#include "rmath_bridge.h"
#ifdef MATHLIB_STANDALONE
#include "../vendor/rmath/Rmath.h"
#else
#include <Rcpp.h>
using namespace R;
#endif

double rmath_digamma(double x) { return digamma(x); }
double rmath_trigamma(double x) { return trigamma(x); }
double rmath_qchisq(double p, double df, int lower_tail, int log_p) { return qchisq(p, df, lower_tail, log_p); }



