#' Summary for SADA estimating equation fits
#'
#' Produces a coefficient table and basic diagnostics for an object returned
#' by \code{sada_ee}.
#'
#' @param object An object of class \code{"sada_ee"}.
#' @param ... Additional arguments, currently ignored.
#'
#' @return
#' An object of class \code{"summary.sada_ee"} containing a coefficient
#' table and basic diagnostics for the corresponding \code{sada_ee} fit.
#'
#' @importFrom stats pnorm
#' @export
summary.sada_ee <- function(object, ...) {

  est <- object$est
  se  <- object$sd
  ci  <- object$ci

  zval <- est / se
  pval <- 2 * pnorm(abs(zval), lower.tail = FALSE)

  coef_tab <- cbind(
    Estimate  = est,
    Std.Error = se,
    `z value` = zval,
    `Pr(>|z|)` = pval,
    CI.lower  = ci[, 1],
    CI.upper  = ci[, 2]
  )

  w_list <- object$w_opt
  w_norm <- sapply(w_list, function(W) sqrt(sum(W^2)))

  out <- list(
    coefficients = coef_tab,
    w_norm      = w_norm,
    converged   = object$bbsolve_sada$convergence == 0,
    iterations  = object$bbsolve_sada$iter,
    call        = object$call
  )
  class(out) <- "summary.sada_ee"
  out
}


#' @importFrom stats printCoefmat
#' @export
print.summary.sada_ee <- function(x, ...) {

  cat("\nCall:\n")
  print(x$call)

  cat("\nSADA estimating equation fit\n")
  cat("Converged:", x$converged,
      "   Iterations:", x$iterations, "\n")

  cat("\nCoefficients:\n")
  printCoefmat(x$coefficients, digits = 4)

  cat("\nWeight matrix norms (per prediction source):\n")
  print(x$w_norm)

  invisible(x)
}



#' Variance covariance matrix for SADA estimating equation fits
#'
#' Extracts the estimated covariance matrix of the SADA estimator from an
#' object returned by \code{sada_ee}.
#'
#' @param object An object of class \code{"sada_ee"} as returned by
#'   \code{sada_ee()}.
#' @param ... Additional arguments, currently ignored.
#'
#' @return A numeric matrix giving the estimated covariance of the SADA
#'   estimator. This is the same matrix that is stored in the \code{vcov}
#'   element of the object.
#'
#' @export
vcov.sada_ee <- function(object, ...) {
  if (is.null(object$vcov)) {
    stop("The supplied object does not contain a covariance matrix component 'vcov'.")
  }
  object$vcov
}
