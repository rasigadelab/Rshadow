#/***********************************************************************************/
#/* SHADOw FRAMEWORK, 2023-2024, <jean-philippe.rasigade@univ-lyon1.fr>             */
#/* SHADOw 2024 by Jean - Philippe Rasigade is licensed under CC BY - NC - ND 4.0 */
#/* License details at https://creativecommons.org/licenses/by-nc-nd/4.0/           */
#/* Alternative private/commercial licensing available upon request                 */
#/***********************************************************************************/

### S4 TENSOR = shadow_tensor ###########################################
setGeneric("tensor", function(x) standardGeneric("tensor"))
setMethod("tensor", c(x = "numeric"), function(x) .shadow_tensor_new(x))
setMethod("tensor", c(x = "array"), function(x) .shadow_tensor_new(x))

setMethod("as.numeric", "shadow_tensor", function(x) .shadow_tensor_as_numeric(x)) 
setMethod("show",       "shadow_tensor", function(object) print(.shadow_tensor_as_numeric(object)))
setMethod("dim",        "shadow_tensor", function(x) attr(as.numeric(x), "dim"))

# Subsetting methods,
# see https://stackoverflow.com/questions/37597266/r-s4-setmethod-distinguish-missing-argument 

setMethod("[", 
    c(x = "shadow_tensor", i = "numeric", j = "numeric", drop = "missing"), 
    function(x, i, j) {
        tensor(as.numeric(x)[i,j])
    }
)
setMethod("[", 
    c(x = "shadow_tensor", i = "missing", j = "numeric", drop = "missing"),
    function(x, i, j) {
        tensor(as.numeric(x)[,j])
    }
)
setMethod("[", 
    c(x = "shadow_tensor", i = "numeric", j = "missing", drop = "missing"),
    function(x, i, j, ..., drop) {
        if(nargs() == 3) {
            tensor(as.numeric(x)[i,])
        } else {
            tensor(as.numeric(x)[i])
        }
    }
)