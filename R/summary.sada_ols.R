#' Summary method for SADA OLS objects
#'
#' Produces a regression-style summary similar to \code{summary.lm}.
#'
#' @param object An object of class \code{"sada_ols"}.
#' @param ... Additional arguments (currently ignored).
#'
#' @return An object of class \code{"summary.sada_ols"} containing
#'   a coefficient table and related quantities.
#' @importFrom stats pnorm printCoefmat
#' @export
summary.sada_ols <- function(object, ...) {

  est <- object$est
  sd  <- object$sd

  zval <- est / sd
  pval <- 2 * (1 - pnorm(abs(zval)))

  coef_table <- cbind(
    Estimate  = est,
    Std.Error = sd,
    `z value` = zval,
    `Pr(>|z|)` = pval
  )

  out <- list(
    coefficients = coef_table,
    ci           = object$ci,
    ci_length    = object$ci_length,
    w_opt        = object$w_opt,
    level        = object$level,
    call         = object$call
  )

  class(out) <- "summary.sada_ols"
  out
}

#' @export
print.summary.sada_ols <- function(x, ...) {
  cat("SADA OLS Regression Results\n\n")
  if (!is.null(x$call)) {
    cat("Call:\n")
    print(x$call)
    cat("\n")
  }
  cat("Coefficients:\n")
  printCoefmat(x$coefficients, P.values = TRUE, has.Pvalue = TRUE)
  invisible(x)
}


#' Variance-covariance matrix for SADA OLS
#'
#' @param object An object of class \code{"sada_ols"}.
#' @param ... Additional arguments (ignored).
#'
#' @return The covariance matrix of the coefficient estimates.
#' @export
vcov.sada_ols <- function(object, ...) {
  object$vcov
}

#' Prediction method for SADA OLS
#'
#' @param object An object of class \code{"sada_ols"}.
#' @param newdata Matrix or data frame of covariates (without intercept
#'   column). If missing, predictions for the labeled design matrix are
#'   not available because the original X is not stored. If \code{newdata}
#'   carries column names that include all fitted covariates, its columns
#'   are selected and reordered by name; otherwise columns are used
#'   positionally and their number must match the fitted covariates.
#' @param ... Additional arguments (ignored).
#'
#' @return Numeric vector of predicted outcomes.
#' @export
predict.sada_ols <- function(object, newdata, ...) {

  if (missing(newdata)) {
    stop("Please provide 'newdata' (covariates without intercept).")
  }

  beta <- object$est
  p_cov <- length(beta) - 1L                 # covariates expected (intercept excluded)
  coef_names <- names(beta)
  ## The intercept is always the first coefficient (sada_ols prepends it), so
  ## the covariate names are simply the remaining ones.
  cov_names <- if (!is.null(coef_names)) coef_names[-1] else character(0)

  ## If newdata carries column names that include all fitted covariates,
  ## select and reorder by name so column order cannot be silently mismatched.
  nd_names <- colnames(newdata)
  if (length(cov_names) == p_cov && !is.null(nd_names) && all(cov_names %in% nd_names)) {
    newdata <- newdata[, cov_names, drop = FALSE]
  }

  X_new <- as.matrix(newdata)
  if (!is.numeric(X_new)) {
    stop("'newdata' must contain only numeric covariate columns.")
  }
  if (ncol(X_new) != p_cov) {
    stop(sprintf(
      paste0("'newdata' has %d column(s) but the model expects %d ",
             "covariate(s) (an intercept is added automatically)."),
      ncol(X_new), p_cov
    ))
  }

  X_new <- cbind(1, X_new)
  drop(X_new %*% beta)
}

