#/***********************************************************************************/
#/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
#/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
#/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
#/* Alternative private/commercial licensing available upon request                 */
#/***********************************************************************************/

#' @title Log-likelihood functions
#' @export
ll_norm <- function(x, mu, sigma) {
  # -1/2 * log(2 pi)
  minus_half_log_2_pi <- -0.918938533204672741780330
  
  z <- (x - mu) / sigma
  ll <- minus_half_log_2_pi - 0.5 * z^2 - log(sigma)
  return(ll)
}

#' @export
squared_diff <- function(x, y) (x - y)^2