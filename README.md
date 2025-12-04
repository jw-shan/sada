# sada

R package for implementing SADA: Safe and Adaptive Aggregation of
Multiple Black-Box Predictions in Semi-Supervised Learning.

## Installation

``` r
install.packages("devtools")
devtools::install_github("jw-shan/sada")
```

## Example: Implement SADA-OLS with two predictions

``` r
library(sada)

## Sample sizes
n <- 100    # labeled
m <- 500   # unlabeled

## True coefficients: intercept and slopes
beta_true <- c(1, 2, -1)

## Covariates for labeled and unlabeled data (without intercept column)
X_labeled <- cbind(rnorm(n), rnorm(n))
X_unlabeled <- cbind(rnorm(m), rnorm(m))

## Labeled outcomes generated from a linear model with noise
linpred_labeled <- beta_true[1] +
  as.numeric(X_labeled %*% beta_true[-1])
Y <- linpred_labeled + rnorm(n, sd = 1)

## Build full covariate matrix (labeled + unlabeled)
X_full <- rbind(X_labeled, X_unlabeled)
N <- n + m

## Underlying linear predictor for all data
linpred_full <- beta_true[1] +
  as.numeric(X_full %*% beta_true[-1])

## Two black box prediction sources with different noise levels
Yhat1_full <- linpred_full + rnorm(N, sd = 0.5)
Yhat2_full <- linpred_full + rnorm(N, sd = 1.5)

## Split into labeled and unlabeled prediction matrices
Yhat_labeled <- cbind(Yhat1_full[1:n], Yhat2_full[1:n])
Yhat_unlabeled <- cbind(Yhat1_full[(n + 1):N], Yhat2_full[(n + 1):N])

## Fit SADA OLS estimator
fit_sada <- sada_ols(
  Y = Y,
  Yhat_labeled = Yhat_labeled,
  Yhat_unlabeled = Yhat_unlabeled,
  X_labeled = X_labeled,
  X_unlabeled = X_unlabeled
)

## Coefficient estimates from SADA (including intercept)
fit_sada$est
```

    ## (Intercept)          X1          X2 
    ##    0.990471    1.942454   -1.164161

``` r
## Standard errors and confidence intervals from SADA
fit_sada$sd
```

    ## (Intercept)          X1          X2 
    ##  0.10816132  0.08060894  0.12429833

``` r
fit_sada$ci
```

    ##                  lower      upper
    ## (Intercept)  0.7784787  1.2024633
    ## X1           1.7844636  2.1004448
    ## X2          -1.4077817 -0.9205412

``` r
## A regression style summary, similar to summary(lm)
summary(fit_sada)
```

    ## SADA OLS Regression Results
    ## 
    ## Call:
    ## sada_ols(Y = Y, Yhat_labeled = Yhat_labeled, Yhat_unlabeled = Yhat_unlabeled, 
    ##     X_labeled = X_labeled, X_unlabeled = X_unlabeled)
    ## 
    ## Coefficients:
    ##              Estimate Std.Error z value  Pr(>|z|)    
    ## (Intercept)  0.990471  0.108161  9.1573 < 2.2e-16 ***
    ## X1           1.942454  0.080609 24.0973 < 2.2e-16 ***
    ## X2          -1.164161  0.124298 -9.3659 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
## Extract the covariance matrix via the vcov method
vcov(fit_sada)
```

    ##               (Intercept)            X1            X2
    ## (Intercept)  0.0116988720 -0.0006347361  0.0005791262
    ## X1          -0.0006347361  0.0064978013 -0.0024637840
    ## X2           0.0005791262 -0.0024637840  0.0154500755

``` r
## Compare with ordinary OLS using labeled data only
## We fit lm on the same covariates, without using any unlabeled data
df_labeled <- data.frame(
  Y = Y,
  X1 = X_labeled[, 1],
  X2 = X_labeled[, 2]
)
fit_lm <- lm(Y ~ X1 + X2, data = df_labeled)

## Coefficients from naive OLS based on labeled data
summary(fit_lm)$coefficients
```

    ##              Estimate Std. Error    t value     Pr(>|t|)
    ## (Intercept)  1.018670 0.11226998   9.073393 1.343189e-14
    ## X1           1.991551 0.11727809  16.981442 8.171864e-31
    ## X2          -1.123504 0.09906548 -11.341027 1.761736e-19

``` r
## Prediction for new covariate values
X_new <- data.frame(X1=c(0,1,2), X2=c(0.5,0,-0.5))
predict(fit_lm, newdata = X_new)
```

    ##         1         2         3 
    ## 0.4569176 3.0102207 5.5635238

``` r
predict(fit_sada, newdata = X_new)
```

    ## [1] 0.4083902 2.9329252 5.4574601
