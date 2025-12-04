#' General SADA estimator for user specified estimating equations
#'
#' Implements a general SADA estimator for parameters defined by estimating
#' equations. The user supplies a score function \code{s_fun}, and the
#' function automatically constructs the SADA adjusted estimating equation
#' using labeled data, unlabeled data and multiple black box predictions.
#' The resulting parameter estimate is obtained by solving the SADA estimating
#' equation via \code{BBsolve}, and a sandwich type variance estimator is
#' used to provide standard errors and confidence intervals.
#' Objects returned by \code{sada_ee} have methods for \code{summary}
#' and \code{vcov}.
#'
#' @param Y Numeric vector of outcomes for the labeled data of length n.
#' @param Yhat_labeled Numeric matrix of size n times K containing black box
#'   predictions for the labeled observations, one column per prediction
#'   source.
#' @param Yhat_unlabeled Numeric matrix of size m times K containing black box
#'   predictions for the unlabeled observations. Must have the same number
#'   of columns as \code{Yhat_labeled}.
#' @param X_labeled Covariate matrix for the labeled data, of size n times q.
#'   This matrix is passed directly to \code{s_fun}.
#' @param X_unlabeled Covariate matrix for the unlabeled data, of size m times q.
#'   Must have the same number of columns as \code{X_labeled}.
#' @param s_fun User supplied function computing individual estimating
#'   equation contributions. It must have the form
#'   \code{s_fun(beta, X, Y)}, where \code{beta} is a numeric parameter
#'   vector of length p, \code{X} is an n times q matrix, and \code{Y} is a
#'   numeric vector of length n. The function must return an n times p matrix
#'   whose i th row is s(x_i, y_i; beta).
#' @param beta_init Optional numeric vector giving the initial value for the
#'   parameter beta with length p. If omitted, the function attempts to infer
#'   the dimension p and uses a zero vector of length p whenever this is
#'   possible. If the dimension cannot be inferred safely, the user is asked
#'   to supply \code{beta_init} explicitly.
#' @param level Confidence level for confidence intervals. Default is 0.95.
#' @param bb_control Optional list of control parameters passed to
#'   \code{BBsolve} via its \code{control} argument.
#'
#' @return An object of class \code{"sada_ee"}, which is a list containing
#' \describe{
#'   \item{est}{Final SADA estimate of the parameter vector.}
#'   \item{sd}{Estimated standard errors for each component of the estimator.}
#'   \item{ci}{Matrix of pointwise confidence intervals for the estimator, with
#'         columns \code{lower} and \code{upper}.}
#'   \item{vcov}{Estimated covariance matrix of the SADA estimator. This is the
#'         same matrix returned by the method \code{vcov.sada_ee}.}
#'   \item{beta_prelim}{Preliminary estimator obtained by solving the labeled only
#'         estimating equation.}
#'   \item{w_opt}{List of length K. Each element is a p times p matrix containing
#'         the optimal SADA weight block for the corresponding prediction source.}
#'   \item{w_opt_full}{Full optimal weight matrix of dimension Kp times p obtained
#'         by stacking all per predictor blocks.}
#'   \item{K}{Number of black box prediction sources.}
#'   \item{bbsolve_prelim}{Selected output of \code{BBsolve} for the preliminary estimator.}
#'   \item{bbsolve_sada}{Selected output of \code{BBsolve} for the SADA estimator.}
#'   \item{call}{The matched function call.}
#' }
#'
#' @examples
#' ## Example: logistic regression via estimating equations
#' ## Target: solve E[ X_i times { Y_i minus expit(X_i^T beta) } ] = 0
#'
#' set.seed(1)
#'
#' n <- 100   # labeled sample size
#' m <- 500   # unlabeled sample size
#'
#' ## Dimension of beta including intercept
#' p <- 3
#'
#' ## Covariates for labeled and unlabeled data
#' X_labeled <- cbind(
#'   1,
#'   matrix(rnorm(n * (p - 1)), nrow = n, ncol = p - 1)
#' )
#' X_unlabeled <- cbind(
#'   1,
#'   matrix(rnorm(m * (p - 1)), nrow = m, ncol = p - 1)
#' )
#'
#' ## True regression coefficients
#' beta_true <- c(-0.5, 1, -1)
#'
#' ## Generate labeled outcomes from a logistic regression model
#' linpred_labeled <- as.numeric(X_labeled %*% beta_true)
#' prob_labeled <- plogis(linpred_labeled)
#' Y <- rbinom(n, size = 1, prob = prob_labeled)
#'
#' ## Build full design matrix labeled plus unlabeled
#' X_full <- rbind(X_labeled, X_unlabeled)
#' N <- n + m
#'
#' ## Underlying linear predictor and probabilities for all data
#' linpred_full <- as.numeric(X_full %*% beta_true)
#' prob_full <- plogis(linpred_full)
#'
#' ## Generate an unobserved true outcome for all N samples
#' Y_full <- rbinom(N, size = 1, prob = prob_full)
#'
#' ## Construct K = 2 black box prediction sources for the probability
#' K <- 2
#' eta1_full <- linpred_full + rnorm(N, sd = 0.5)
#' eta2_full <- linpred_full + rnorm(N, sd = 1.0)
#' Yhat1_full <- plogis(eta1_full)
#' Yhat2_full <- plogis(eta2_full)
#'
#' ## Here the black box predictions are probabilities
#' ## For sada_ee we treat them as pseudo outcomes in [0, 1]
#'
#' Yhat_labeled   <- cbind(Yhat1_full[1:n],     Yhat2_full[1:n])
#' Yhat_unlabeled <- cbind(Yhat1_full[(n+1):N], Yhat2_full[(n+1):N])
#'
#' ## Score function for logistic regression:
#' ## s(beta, X, Y)_i = X_i times ( Y_i minus expit(X_i^T beta) )
#' s_fun <- function(beta, X, Y) {
#'   X <- as.matrix(X)
#'   linpred <- as.numeric(X %*% beta)
#'   mu <- plogis(linpred)
#'   X * (Y - mu)
#' }
#'
#' ## Initial value for beta
#' beta_init <- rep(0, p)
#'
#' fit_logit <- sada_ee(
#'   Y = Y,
#'   Yhat_labeled = Yhat_labeled,
#'   Yhat_unlabeled = Yhat_unlabeled,
#'   X_labeled = X_labeled,
#'   X_unlabeled = X_unlabeled,
#'   s_fun = s_fun,
#'   beta_init = beta_init,
#'   level = 0.95
#' )
#'
#' ## SADA estimate for the logistic regression coefficients
#' fit_logit$est
#'
#' ## Standard errors and confidence intervals
#' fit_logit$sd
#' fit_logit$ci
#'
#' ## Summary and covariance matrix
#' summary(fit_logit)
#' vcov(fit_logit)
#'
#' ## Optimal SADA weight matrices, one per prediction source
#' fit_logit$w_opt$pred1
#' fit_logit$w_opt$pred2
#'
#' @importFrom BB BBsolve
#' @importFrom stats cov qnorm
#' @importFrom numDeriv jacobian
#' @export
sada_ee <- function(Y,
                    Yhat_labeled,
                    Yhat_unlabeled,
                    X_labeled,
                    X_unlabeled,
                    s_fun,
                    beta_init = NULL,
                    level = 0.95,
                    bb_control = list()) {

  call <- match.call()

  ## Basic checks
  if (length(Y) != NROW(X_labeled)) {
    stop("Dimensions of Y and X_labeled do not match.")
  }
  if (length(Y) != NROW(Yhat_labeled)) {
    stop("Length of Y must match the number of rows in Yhat_labeled.")
  }
  if (NCOL(Yhat_labeled) != NCOL(Yhat_unlabeled)) {
    stop("Number of columns in Yhat_labeled and Yhat_unlabeled must match.")
  }
  if (NROW(X_unlabeled) != NROW(Yhat_unlabeled)) {
    stop("Number of rows in X_unlabeled and Yhat_unlabeled must match.")
  }
  if (NCOL(X_labeled) != NCOL(X_unlabeled)) {
    stop("X_labeled and X_unlabeled must have the same number of columns.")
  }
  if (!is.numeric(level) || level <= 0 || level >= 1) {
    stop("level must be a number strictly between 0 and 1.")
  }

  ## Convert to matrices
  Y <- as.numeric(Y)
  X_labeled <- as.matrix(X_labeled)
  X_unlabeled <- as.matrix(X_unlabeled)
  Yhat_labeled <- as.matrix(Yhat_labeled)
  Yhat_unlabeled <- as.matrix(Yhat_unlabeled)

  n <- NROW(X_labeled)
  m <- NROW(X_unlabeled)
  N <- n + m
  if (m <= 0) {
    stop("There must be at least one unlabeled observation.")
  }
  K <- NCOL(Yhat_labeled)

  ## Determine parameter dimension p safely
  if (!is.null(beta_init)) {
    beta_try <- as.numeric(beta_init)
    s0 <- tryCatch(
      s_fun(beta_try, X_labeled, Y),
      error = function(e) {
        stop(
          "s_fun produced an error when evaluated at the supplied beta_init.\n",
          "Error message: ", conditionMessage(e)
        )
      }
    )
    s0 <- as.matrix(s0)
    if (NROW(s0) != n) {
      stop("s_fun must return an n times p matrix, rows correspond to observations.")
    }
    p <- NCOL(s0)
    if (p != length(beta_try)) {
      stop(
        "The supplied beta_init has length = ", length(beta_try),
        " but s_fun returns a score of dimension p = ", p, ".\n",
        "Please supply beta_init with the correct length."
      )
    }
  } else {
    beta_try <- rep(0, NCOL(X_labeled))
    s0 <- tryCatch(
      s_fun(beta_try, X_labeled, Y),
      error = function(e) {
        stop(
          "Could not automatically infer beta_init.\n",
          "A trial call s_fun(beta_try, X_labeled, Y) with beta_try of length ",
          length(beta_try), " produced an error.\n",
          "Error message: ", conditionMessage(e), "\n\n",
          "Please specify beta_init manually with the correct dimension."
        )
      }
    )
    s0 <- as.matrix(s0)
    if (NROW(s0) != n) {
      stop("s_fun must return an n times p matrix, rows correspond to observations.")
    }
    p <- NCOL(s0)
    if (p != length(beta_try)) {
      stop(
        "Could not automatically infer beta_init.\n",
        "Guessed length for beta was ", length(beta_try),
        " but s_fun returned a score of dimension p = ", p, ".\n",
        "Please specify beta_init manually with length = ", p, "."
      )
    }
    beta_init <- rep(0, p)
  }

  ## 1. Preliminary estimator: labeled only estimating equation
  ee_supervised <- function(beta) {
    s_mat <- s_fun(beta, X_labeled, Y)
    colMeans(s_mat)
  }

  ans_prelim <- BBsolve(
    par = beta_init,
    fn  = ee_supervised,
    control = bb_control,
    quiet = TRUE
  )
  beta_prelim <- as.numeric(ans_prelim$par)

  ## 2. Optimal weight matrix using S built from s_fun with Yhat
  S_blocks <- lapply(seq_len(K), function(k) {
    s_fun(beta_prelim, X_labeled, Yhat_labeled[, k])
  })
  S_l <- do.call(cbind, S_blocks)
  S_l <- as.matrix(S_l)
  if (NROW(S_l) != n) {
    stop("Internal error, S_l must have n rows.")
  }
  if (NCOL(S_l) != K * p) {
    stop("Internal error, S_l must have K times p columns.")
  }

  s_l_pre <- s_fun(beta_prelim, X_labeled, Y)
  s_l_pre <- as.matrix(s_l_pre)
  if (NCOL(s_l_pre) != p) {
    stop("s_fun must return an n times p matrix consistently.")
  }

  A <- crossprod(S_l) / n               # approx E[ S S^T ]
  B <- crossprod(S_l, s_l_pre) / n      # approx E[ S s^T ]
  w_opt_mat <- solve(A, B)              # (Kp times p)

  ## 3. SADA estimating equation using fixed w_opt_mat
  ee_sada <- function(beta) {
    sY <- s_fun(beta, X_labeled, Y)
    g_super <- colMeans(sY)

    delta <- numeric(K * p)
    for (k in seq_len(K)) {
      s_l_k <- s_fun(beta, X_labeled,   Yhat_labeled[, k])
      s_u_k <- s_fun(beta, X_unlabeled, Yhat_unlabeled[, k])

      avg_l_k <- colMeans(as.matrix(s_l_k))
      avg_u_k <- colMeans(as.matrix(s_u_k))

      idx <- ((k - 1) * p + 1):(k * p)
      delta[idx] <- avg_u_k - avg_l_k
    }

    as.numeric(g_super + t(w_opt_mat) %*% delta)
  }

  ans_sada <- BBsolve(
    par = beta_prelim,
    fn  = ee_sada,
    control = bb_control,
    quiet = TRUE
  )
  beta_sada <- as.numeric(ans_sada$par)

  #### 4. Variance estimation for SADA EE ####

  ## Scores at beta_sada on labeled data
  sY_hat <- s_fun(beta_sada, X_labeled, Y)
  sY_hat <- as.matrix(sY_hat)

  ## Stacked S(X, Yhat; beta_sada) on labeled data: n times (Kp)
  S_list_hat <- lapply(seq_len(K), function(k) {
    s_fun(beta_sada, X_labeled, Yhat_labeled[, k])
  })
  S_l_hat <- do.call(cbind, S_list_hat)
  S_l_hat <- as.matrix(S_l_hat)

  ## Sample moments for s and S
  mean_s <- colMeans(sY_hat)      # length p
  mean_S <- colMeans(S_l_hat)     # length Kp

  ## E[s s^T] and E[S S^T]
  E_ssT  <- crossprod(sY_hat)  / n
  E_SS_T <- crossprod(S_l_hat) / n

  ## Var(s) and Var(S)
  Var_s <- E_ssT  - tcrossprod(mean_s)
  Var_S <- E_SS_T - tcrossprod(mean_S)

  ## E[S s^T] and E[s S^T]
  E_SsT <- crossprod(S_l_hat, sY_hat) / n   # Kp times p
  E_sST <- t(E_SsT)                         # p times Kp

  ## Regularize Var(S) for numerical stability
  lambda_S <- 1e-6
  Var_S_reg <- Var_S + lambda_S * diag(nrow(Var_S))

  ## Sigma_opt according to the theoretical formula
  ## Sigma_opt = Var(s) minus (N-n)/N * E[s S^T] Var(S)^{-1} E[S s^T]
  factor_NS <- (N - n) / N
  tmp <- solve(Var_S_reg, E_SsT)           # (Var(S) + λI)^{-1} E[S s^T]
  Sigma_opt <- Var_s - factor_NS * (E_sST %*% tmp)

  ## Numerical Jacobian H of the supervised EE at beta_sada
  ## H = d / d beta E[s(X, Y; beta)] at beta_sada
  H <- jacobian(ee_supervised, beta_sada)

  ## Regularize H for numerical stability
  lambda_H <- 1e-6
  H_reg <- H + lambda_H * diag(p)

  ## H^{-1} via solve(H_reg, I)
  Hinv  <- solve(H_reg, diag(p))
  HinvT <- t(Hinv)

  vcov_beta <- (1 / n) * Hinv %*% Sigma_opt %*% HinvT
  sd_beta <- sqrt(diag(vcov_beta))

  alpha <- 1 - level
  z <- qnorm(1 - alpha / 2)
  ci_lower <- beta_sada - z * sd_beta
  ci_upper <- beta_sada + z * sd_beta
  ci_mat <- cbind(lower = ci_lower, upper = ci_upper)

  ## 5. Split w_opt_mat by prediction into a list
  w_list <- vector("list", K)
  for (k in seq_len(K)) {
    idx <- ((k - 1) * p + 1):(k * p)
    w_list[[k]] <- w_opt_mat[idx, , drop = FALSE]
  }
  names(w_list) <- paste0("pred", seq_len(K))

  bb_prelim_info <- ans_prelim[c("par", "convergence", "message", "iter", "feval")]
  bb_sada_info   <- ans_sada[c("par", "convergence", "message", "iter", "feval")]

  out <- list(
    est            = beta_sada,
    sd             = sd_beta,
    ci             = ci_mat,
    vcov           = vcov_beta,
    beta_prelim    = beta_prelim,
    w_opt          = w_list,
    w_opt_full     = w_opt_mat,
    K              = K,
    bbsolve_prelim = bb_prelim_info,
    bbsolve_sada   = bb_sada_info,
    call           = call
  )
  class(out) <- "sada_ee"
  out
}
