#/***********************************************************************************/
#/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
#/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
#/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
#/* Alternative private/commercial licensing available upon request                 */
#/***********************************************************************************/

### S4 SPY = shadow_spy #################################################
setGeneric("spy", function(x, tape) standardGeneric("spy"))
setMethod("spy", c(x = "shadow_tensor", tape = "shadow_tape"), function(x, tape) .shadow_spy_new(x, tape))
setMethod("spy", c(x = "numeric", tape = "shadow_tape"), function(x, tape) .shadow_spy_new(tensor(x), tape))
setMethod("spy", c(x = "array", tape = "shadow_tape"), function(x, tape) .shadow_spy_new(tensor(x), tape))

setMethod("as.numeric", "shadow_spy", function(x) .shadow_spy_as_numeric(x))
setMethod("show",       "shadow_spy", function(object) print(.shadow_spy_as_numeric(object)))
setMethod("dim",        "shadow_spy", function(x) attr(.shadow_spy_as_numeric(x), "dim"))

setGeneric("read", function(spy, trace) standardGeneric("read"))
setMethod("read",
    c(spy = "shadow_spy", trace = "shadow_trace"),
    function(spy, trace) {
        .shadow_spy_read_on_trace(spy, trace)
    }
)

setMethod("[", 
    c(x = "shadow_spy", i = "numeric", j = "numeric"),
    function(x, i, j) {
        if(length(i) != 1 | length(j) != 1) stop("Spy subsetting can only extract a single element")
        .shadow_spy_subset_i_j(x, i, j)
    }
)
setMethod("[",
    c(x = "shadow_spy", i = "numeric", j = "missing", drop = "missing"),
    function(x, i, j, ..., drop) {
        if(nargs() == 3) {
            stop("Spy subsetting can only extract a single element")
        } else {
            .shadow_spy_subset_i(x, i)
        }
    }
)

