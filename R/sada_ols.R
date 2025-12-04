#' SADA estimator for linear regression coefficients
#'
#' @description
#' Implements SADA for the linear model \code{Y ~ X} using labeled and
#' unlabeled data with multiple black-box outcome predictions. The covariate
#' matrices \code{X_labeled} and \code{X_unlabeled} should NOT contain an
#' intercept column; a column of ones is added internally.
#'
#' @details
#' The output of \code{sada_ols} behaves similarly to a fitted
#' \code{\link[stats]{lm}} object, and several S3 methods are provided:
#'
#' \describe{
#'
#'   \item{\code{summary(fit)}}{
#'     Produces a regression-style summary table with estimates, standard errors,
#'     z-statistics, and p-values. The printed output mirrors the structure of
#'     \code{summary.lm}. Confidence intervals are also included in the
#'     returned object.
#'   }
#'
#'   \item{\code{predict(fit, newdata)}}{
#'     Generates predictions for new covariate values. The argument
#'     \code{newdata} must be a matrix or data frame \emph{without} an intercept
#'     column; a leading column of ones is added automatically. Predictions are
#'     computed as \eqn{X_{\text{new}} \hat\beta_{\text{SADA}}}.
#'   }
#'
#'   \item{\code{vcov(fit)}}{
#'     Returns the estimated covariance matrix of the SADA coefficient
#'     estimator \eqn{\hat\beta_{\text{SADA}}}. The diagonal entries match the
#'     squared standard errors reported in \code{summary(fit)}.
#'   }
#'
#'   \item{\code{fit$w_opt}}{
#'     A list of \eqn{K} matrices containing the optimal SADA weights applied
#'     to each black-box prediction source. Each matrix has dimension
#'     \eqn{p \times p} and corresponds to the contribution of that prediction
#'     to the calibration of \eqn{X^\top Y} and \eqn{X^\top X}. These weights
#'     determine how SADA adaptively borrows information from unlabeled data.
#'   }
#'
#' }
#'
#' A typical workflow is:
#'
#' \enumerate{
#'   \item Fit the SADA estimator: \code{fit <- sada_ols(...)}.
#'   \item Inspect estimated coefficients using \code{summary(fit)}.
#'   \item Extract standard errors or covariance matrix if needed.
#'   \item Use \code{predict(fit, newdata)} for out-of-sample prediction.
#'   \item Examine \code{fit$w_opt} to understand the influence of each
#'         prediction source.
#' }
#'
#' @param Y Numeric vector of outcomes for the labeled data.
#' @param Yhat_labeled Numeric matrix of predictions for labeled samples.
#' @param Yhat_unlabeled Numeric matrix of predictions for unlabeled samples.
#' @param X_labeled Covariate matrix for labeled data (no intercept column).
#' @param X_unlabeled Covariate matrix for unlabeled data (no intercept column).
#' @param level Confidence level (default 0.95).
#'
#' @return An object of class \code{"sada_ols"} which is a list with:
#' \describe{
#'   \item{est}{Estimated regression coefficients (including intercept).}
#'   \item{sd}{Standard errors of each coefficient.}
#'   \item{ci}{Matrix with two columns \code{lower} and \code{upper}
#'             giving the confidence interval for each coefficient.}
#'   \item{ci_length}{Length of each confidence interval.}
#'   \item{w_opt}{List of optimal SADA weight matrices (one per predictor).}
#'   \item{vcov}{Estimated covariance matrix of the coefficients.}
#'   \item{level}{Confidence level used for the intervals.}
#'   \item{call}{The matched function call.}
#' }
#'
#' @examples
#' set.seed(123)
#'
#' ## Example: compare SADA OLS with ordinary OLS using labeled data only
#'
#' ## Sample sizes
#' n <- 80    # labeled
#' m <- 200   # unlabeled
#'
#' ## True coefficients: intercept and slopes
#' beta_true <- c(1, 2, -1)
#'
#' ## Covariates for labeled and unlabeled data (without intercept column)
#' X_labeled <- cbind(rnorm(n), rnorm(n))
#' X_unlabeled <- cbind(rnorm(m), rnorm(m))
#'
#' ## Labeled outcomes generated from a linear model with noise
#' linpred_labeled <- beta_true[1] +
#'   as.numeric(X_labeled %*% beta_true[-1])
#' Y <- linpred_labeled + rnorm(n, sd = 1)
#'
#' ## Build full covariate matrix (labeled + unlabeled)
#' X_full <- rbind(X_labeled, X_unlabeled)
#' N <- n + m
#'
#' ## Underlying linear predictor for all data
#' linpred_full <- beta_true[1] +
#'   as.numeric(X_full %*% beta_true[-1])
#'
#' ## Two black box prediction sources with different noise levels
#' Yhat1_full <- linpred_full + rnorm(N, sd = 0.5)
#' Yhat2_full <- linpred_full + rnorm(N, sd = 1.5)
#'
#' ## Split into labeled and unlabeled prediction matrices
#' Yhat_labeled <- cbind(Yhat1_full[1:n], Yhat2_full[1:n])
#' Yhat_unlabeled <- cbind(Yhat1_full[(n + 1):N], Yhat2_full[(n + 1):N])
#'
#' ## Fit SADA OLS estimator
#' ## X_labeled and X_unlabeled do not contain an intercept
#' ## A column of ones is added inside sada_ols
#' fit_sada <- sada_ols(
#'   Y = Y,
#'   Yhat_labeled = Yhat_labeled,
#'   Yhat_unlabeled = Yhat_unlabeled,
#'   X_labeled = X_labeled,
#'   X_unlabeled = X_unlabeled
#' )
#'
#' ## Coefficient estimates from SADA (including intercept)
#' fit_sada$est
#'
#' ## Standard errors and confidence intervals from SADA
#' fit_sada$sd
#' fit_sada$ci
#'
#' ## A regression style summary, similar to summary(lm)
#' summary(fit_sada)
#'
#' ## Extract the covariance matrix via the vcov method
#' vcov(fit_sada)
#'
#' ## Compare with ordinary OLS using labeled data only
#' ## We fit lm on the same covariates, without using any unlabeled data
#' df_labeled <- data.frame(
#'   Y = Y,
#'   X1 = X_labeled[, 1],
#'   X2 = X_labeled[, 2]
#' )
#' fit_lm <- lm(Y ~ X1 + X2, data = df_labeled)
#'
#' ## Coefficients from naive OLS based on labeled data
#' coef(fit_lm)
#'
#' ## Prediction for new covariate values
#' ## newdata must be a matrix or data frame without an intercept column
#' X_new <- cbind(rnorm(5), rnorm(5))
#'
#' ## SADA predictions
#' pred_sada <- predict(fit_sada, newdata = X_new)
#'
#' @importFrom stats lm qnorm cov coef residuals
#' @export
sada_ols <- function(Y,
                     Yhat_labeled,
                     Yhat_unlabeled,
                     X_labeled,
                     X_unlabeled,
                     level = 0.95) {

  call <- match.call()

  ## Basic checks
  if (length(Y) != NROW(X_labeled)) {
    stop("Dimensions of Y and X_labeled do not match.")
  }
  if (length(Y) != NROW(Yhat_labeled)) {
    stop("Length of Y must match the number of rows in Yhat_labeled.")
  }
  if (NCOL(Yhat_labeled) != NCOL(Yhat_unlabeled)) {
    stop("Number of predictors in Yhat_labeled and Yhat_unlabeled must match.")
  }
  if (NROW(X_unlabeled) != NROW(Yhat_unlabeled)) {
    stop("Number of rows in X_unlabeled and Yhat_unlabeled must match.")
  }
  if (NCOL(X_labeled) != NCOL(X_unlabeled)) {
    stop("X_labeled and X_unlabeled must have the same number of columns.")
  }
  if (!is.numeric(level) || level <= 0 || level >= 1) {
    stop("level must be between 0 and 1.")
  }

  ## Convert to matrices
  Y <- as.numeric(Y)
  X_labeled <- as.matrix(X_labeled)
  X_unlabeled <- as.matrix(X_unlabeled)
  Yhat_labeled <- as.matrix(Yhat_labeled)
  Yhat_unlabeled <- as.matrix(Yhat_unlabeled)

  ## Add intercept column automatically
  X_labeled <- cbind(1, X_labeled)
  X_unlabeled <- cbind(1, X_unlabeled)

  ## Dimensions
  n <- NROW(X_labeled)
  m <- NROW(X_unlabeled)
  if (m <= 0) {
    stop("There must be at least one unlabeled observation.")
  }
  N <- n + m
  p <- NCOL(X_labeled)
  K <- NCOL(Yhat_labeled)

  ## Give coefficient names
  x_names <- colnames(X_labeled)
  if (is.null(x_names)) {
    x_names <- c("(Intercept)", paste0("X", seq_len(p - 1)))
    colnames(X_labeled) <- x_names
    colnames(X_unlabeled) <- x_names
  }

  ## Combine full X and prediction matrices
  X_full <- rbind(X_labeled, X_unlabeled)
  Yhat_full <- rbind(Yhat_labeled, Yhat_unlabeled)

  #### 1. Naive OLS estimator using labeled data via lm()
  df_labeled <- data.frame(Y = Y, X_labeled)
  fit_naive <- lm(Y ~ . - 1, data = df_labeled)
  beta_naive <- coef(fit_naive)
  resid_l <- residuals(fit_naive)
  linpred_full <- as.numeric(X_full %*% beta_naive)

  #### 2. Build calibration matrices for SADA
  s_l <- X_labeled * resid_l
  pred_resid <- Yhat_full - linpred_full

  calS <- do.call(
    cbind,
    lapply(seq_len(K), function(k) X_full * pred_resid[, k])
  )
  calS_l <- calS[1:n, , drop = FALSE]

  cov_calS_calS <- cov(calS_l, calS_l)
  cov_calS_s <- cov(calS_l, s_l)
  cov_s_s <- cov(s_l, s_l)
  cov_s_calS <- cov(s_l, calS_l)

  #### 3. Optimal weights
  w_opt_mat <- solve(cov_calS_calS, cov_calS_s) * (m / N)

  #### 4. SADA adjusted estimating equations
  XX_l <- crossprod(X_labeled)
  XX_u <- crossprod(X_unlabeled)
  XY_l <- crossprod(X_labeled, Y)

  xy_stack_l <- matrix(
    as.vector(crossprod(X_labeled, Yhat_labeled)),
    ncol = 1
  )
  xy_stack_u <- matrix(
    as.vector(crossprod(X_unlabeled, Yhat_unlabeled)),
    ncol = 1
  )

  xy_sada <- XY_l / n +
    t(w_opt_mat) %*% xy_stack_u / m -
    t(w_opt_mat) %*% xy_stack_l / n

  XX_stack_l <- do.call(rbind, replicate(K, XX_l, simplify = FALSE))
  XX_stack_u <- do.call(rbind, replicate(K, XX_u, simplify = FALSE))

  xx_sada <- XX_l / n +
    t(w_opt_mat) %*% XX_stack_u / m -
    t(w_opt_mat) %*% XX_stack_l / n

  beta_sada <- solve(xx_sada, xy_sada)
  beta_sada <- as.numeric(beta_sada)
  names(beta_sada) <- x_names

  #### 5. Variance of SADA estimator
  H <- crossprod(X_full) / N
  sd_int <- cov_s_s / n -
    cov_s_calS %*% solve(cov_calS_calS, cov_calS_s) * (1 / n - 1 / N)

  vcov_beta <- solve(H, sd_int) %*% solve(H)
  colnames(vcov_beta) <- rownames(vcov_beta) <- x_names
  sd_beta <- sqrt(diag(vcov_beta))
  names(sd_beta) <- x_names

  #### 6. Confidence intervals
  z <- qnorm(1 - (1 - level) / 2)
  ci_lower <- beta_sada - z * sd_beta
  ci_upper <- beta_sada + z * sd_beta
  ci_length <- ci_upper - ci_lower
  ci_mat <- cbind(lower = ci_lower, upper = ci_upper)
  rownames(ci_mat) <- x_names
  names(ci_length) <- x_names

  #### 7. Split w_opt into matrices, one per predictor
  w_list <- vector("list", K)
  for (k in seq_len(K)) {
    idx <- ((k - 1) * p + 1):(k * p)
    w_list[[k]] <- w_opt_mat[idx, , drop = FALSE]
    rownames(w_list[[k]]) <- x_names
    colnames(w_list[[k]]) <- x_names
  }
  names(w_list) <- paste0("pred", seq_len(K))

  out <- list(
    est       = beta_sada,
    sd        = sd_beta,
    ci        = ci_mat,
    ci_length = ci_length,
    w_opt     = w_list,
    vcov      = vcov_beta,
    level     = level,
    call      = call
  )
  class(out) <- "sada_ols"
  out
}
