/***********************************************************************************/
/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
/* SHADOw © 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
/* Alternative private/commercial licensing available upon request                 */
/***********************************************************************************/
#pragma once

/* ADAPTED FROM PUBLIC DOMAIN JAVA FMIN
Brent's minimum finding procedure
PUBLIC DOMAIN
Accessed (in Java):
http://www1.fpl.fs.fed.us/Fmin.java (2015 access)

Accessed (in Java):
https://www.seas.upenn.edu/~eeaton/software/Utils/javadoc/optimization/Fmin.html (2024 access)

Modified for performance, C++ specifics, and SHADOw-C integration
by Jean-Philippe Rasigade as part of the SHADOw-C library.
*/

/* ORIGINAL COPYRIGHT NOTICE
Fmin.java copyright claim:

	This software is based on the public domain fmin routine.
	The FORTRAN version can be found at

	www.netlib.org

	This software was translated from the FORTRAN version
	to Java by a US government employee on official time.
	Thus this software is also in the public domain.


	The translator's mail address is:

	Steve Verrill
	USDA Forest Products Laboratory
	1 Gifford Pinchot Drive
	Madison, Wisconsin
	53705


	The translator's e-mail address is:

	steve@www1.fpl.fs.fed.us


***********************************************************************

DISCLAIMER OF WARRANTIES:

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
THE TRANSLATOR DOES NOT WARRANT, GUARANTEE OR MAKE ANY REPRESENTATIONS
REGARDING THE SOFTWARE OR DOCUMENTATION IN TERMS OF THEIR CORRECTNESS,
RELIABILITY, CURRENTNESS, OR OTHERWISE. THE ENTIRE RISK AS TO
THE RESULTS AND PERFORMANCE OF THE SOFTWARE IS ASSUMED BY YOU.
IN NO CASE WILL ANY PARTY INVOLVED WITH THE CREATION OR DISTRIBUTION
OF THE SOFTWARE BE LIABLE FOR ANY DAMAGE THAT MAY RESULT FROM THE USE
OF THIS SOFTWARE.

Sorry about that.

***********************************************************************


History:

Date        Translator        Changes

3/24/98     Steve Verrill     Translated
*/


/* Generic Brent minimizer */
struct brent_opt_output_t {
	/* Input at the minimum */
	double min = 0.0;
	/* Output at the minimum */
	double objective = 0.0;
	/* No. of function evaluations */
	size_t n_eval = 0;
};

/* Minimize (default) or maximize a function. Here T is a functor with member 'double operator()(double)' */
template< typename T >
brent_opt_output_t brent_optimize(
	T& functor,
	double left, double right,
	bool maximize = false, 
	double tol = sqrt(std::numeric_limits< double >::epsilon())
) {

	static const double DBL_EPSILON_GCC = std::numeric_limits<double>::epsilon();
	static const double DBL_EPSILON_SQRT = 1.490116119384765625e-08;
	static const double GOLDEN_SQRT_INV = 0.3819660112501050974743;

	/* EARLY EXIT */
	if (fabs(right - left) <= DBL_EPSILON_GCC) {
		return { left, functor(left), 0 };
	}

	/* Local variables */
	double a, b, d, e, p, q, r, u, v, w, x;
	double t2, fu, fv, fw, fx, xm, eps, tol1, tol3;

	tol1 = DBL_EPSILON_GCC + 1.;
	eps = DBL_EPSILON_SQRT;

	a = left;
	b = right;
	v = a + GOLDEN_SQRT_INV * (b - a);
	w = v;
	x = v;

	d = 0.;
	e = 0.;

	fx = functor(x);
	if (maximize == true) fx = -fx;
	fv = fx;
	fw = fx;
	tol3 = tol / 3.;

	size_t n_iter = 0;
	while (true) {
		xm = (a + b) * .5;
		tol1 = eps * fabs(x) + tol3;
		t2 = tol1 * 2.;

		// Convergence reached ?
		if (fabs(x - xm) <= t2 - (b - a) * .5) break;
		p = 0.;
		q = 0.;
		r = 0.;
		if (fabs(e) > tol1) {
			// Parabolic fitting
			r = (x - w) * (fx - fv);
			q = (x - v) * (fx - fw);
			p = (x - v) * q - (x - w) * r;
			q = (q - r) * 2.;
			if (q > 0.) p = -p; else q = -q;
			r = e;
			e = d;
		}

		if (fabs(p) >= fabs(q * .5 * r) ||
			p <= q * (a - x) || p >= q * (b - x)) {
			// Golden-section search step
			if (x < xm) e = b - x; else e = a - x;
			d = GOLDEN_SQRT_INV * e;
		}
		else {
			// Parabolic search step
			d = p / q;
			u = x + d;

			// Check we're far enough from the bracket boundaries
			if (u - a < t2 || b - u < t2) {
				d = tol1;
				if (x >= xm) d = -d;
			}
		}

		/* f must not be evaluated too close to x */

		if (fabs(d) >= tol1)
			u = x + d;
		else if (d > 0.)
			u = x + tol1;
		else
			u = x - tol1;

		fu = functor(u);
		if (maximize == true) fu = -fu;

		// Housekeeping
		if (fu <= fx) {
			if (u < x) b = x; else a = x;
			v = w;    w = x;   x = u;
			fv = fw; fw = fx; fx = fu;
		}
		else {
			if (u < x) a = u; else b = u;
			if (fu <= fw || w == x) {
				v = w; fv = fw;
				w = u; fw = fu;
			}
			else if (fu <= fv || v == x || v == w) {
				v = u; fv = fu;
			}
		}

		n_iter++;
	}

	return { x, functor(x), n_iter};
}