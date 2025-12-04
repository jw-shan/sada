
#' SADA estimator for the outcome mean
#'
#' Implements SADA for the marginal outcome mean using labeled and unlabeled
#' data with multiple black-box predictions. A normal-approximation confidence
#' interval is also returned.
#'
#' @param Y Numeric vector of outcomes for the labeled data.
#' @param Yhat_labeled Numeric matrix or data frame of predictions for the
#'   labeled observations. Each column corresponds to a black-box predictor.
#' @param Yhat_unlabeled Numeric matrix or data frame of predictions for the
#'   unlabeled observations. Must have the same number of columns as
#'   \code{Yhat_labeled}.
#' @param level Confidence level for the interval. Must be a number between
#'   zero and one. The default is 0.95.
#'
#' @return A list containing:
#' \describe{
#'   \item{est}{Point estimate of the outcome mean.}
#'   \item{sd}{Estimated standard deviation of the estimator.}
#'   \item{ci}{A numeric vector of length two giving the lower and upper
#'             bounds of a confidence interval at the requested \code{level}.}
#'   \item{ci_length}{Length of the confidence interval.}
#'   \item{w_opt}{Estimated optimal aggregation weight vector.}
#' }
#'
#' @examples
#' set.seed(123)
#'
#' # Parameters
#' theta <- 0.5
#' gamma <- 0.7
#' n <- 60
#' N <- 200
#'
#' # Generate outcome Y for the labeled data
#' Y_gt <- rnorm(N, mean = theta, sd = 1)
#' Y <- Y_gt[1:n]
#'
#' # Generate two black-box predictions for all data
#' eps1 <- rnorm(N, 0, 1)
#' eps2 <- rnorm(N, 0, 1)
#'
#' yh1 <- gamma * Y_gt + (1 - gamma) * eps1
#' yh2 <- (1 - gamma) * Y_gt + gamma * eps2
#'
#' Yhat_full <- cbind(yh1, yh2)
#'
#' # Split into labeled and unlabeled predictions
#' Yhat_labeled <- Yhat_full[1:n, ]
#' Yhat_unlabeled <- Yhat_full[(n + 1):N, ]
#'
#' # Apply SADA estimator
#' result <- sada_mean(Y, Yhat_labeled, Yhat_unlabeled)
#' result$est
#' result$ci
#'
#' @importFrom stats cov qnorm
#' @export
sada_mean <- function(Y,Yhat_labeled,Yhat_unlabeled, level = 0.95){


  # Basic input checks
  if (length(Y) != NROW(Yhat_labeled)) {
    stop("Length of Y must match the number of rows in Yhat_labeled.")
  }
  if (NCOL(Yhat_labeled) != NCOL(Yhat_unlabeled)) {
    stop("Yhat_labeled and Yhat_unlabeled must have the same number of columns.")
  }
  if (!is.numeric(level) || level <= 0 || level >= 1) {
    stop("level must be a numeric value between 0 and 1.")
  }


  #  Convert inputs to appropriate types
  Y <- as.numeric(Y)
  Yhat_labeled <- as.matrix(Yhat_labeled)
  Yhat_unlabeled <- as.matrix(Yhat_unlabeled)

  # Sample sizes
  n <- NROW(Yhat_labeled)              # number of labeled observations
  N <- n + NROW(Yhat_unlabeled)        # total number of observations

  # Combine predictions from labeled and unlabeled data
  yh <- rbind(Yhat_labeled, Yhat_unlabeled)

  # SADA weights for aggregating black-box predictions
  w.opt <- solve(cov(yh,yh),cov(Yhat_labeled,Y)) * (N-n)/N

  # Point estimate of the outcome mean
  est.sada <- mean(Y) +
    as.numeric(t(w.opt) %*% (colMeans(Yhat_unlabeled) - colMeans(Yhat_labeled)))

  # Estimated standard deviation of the estimator
  sd.sada <- sqrt(as.numeric(
      cov(Y,Y)/n - cov(Y,Yhat_labeled)%*%solve(cov(Yhat_labeled,Yhat_labeled),cov(Yhat_labeled,Y))*(1/n-1/N)
    ))

  # Construct confidence interval
  alpha <- 1 - level
  z <- qnorm(1 - alpha / 2)

  ci.lower <- est.sada - z * sd.sada
  ci.upper <- est.sada + z * sd.sada
  ci.length <- ci.upper - ci.lower

  list(
    est       = est.sada,
    sd        = sd.sada,
    ci        = c(ci.lower, ci.upper),
    ci_length = ci.length,
    w_opt     = as.numeric(w.opt)
  )
}





