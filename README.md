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
    ##   0.9100656   1.8614387  -0.9123843

``` r
## Standard errors and confidence intervals from SADA
fit_sada$sd
```

    ## (Intercept)          X1          X2 
    ##  0.09581569  0.09346693  0.09241448

``` r
fit_sada$ci
```

    ##                  lower      upper
    ## (Intercept)  0.7222703  1.0978609
    ## X1           1.6782469  2.0446305
    ## X2          -1.0935133 -0.7312552

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
    ## (Intercept)  0.910066  0.095816  9.4981 < 2.2e-16 ***
    ## X1           1.861439  0.093467 19.9155 < 2.2e-16 ***
    ## X2          -0.912384  0.092414 -9.8727 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
## Extract the covariance matrix via the vcov method
vcov(fit_sada)
```

    ##               (Intercept)            X1            X2
    ## (Intercept)  0.0091806468 -0.0005133307  0.0020493098
    ## X1          -0.0005133307  0.0087360677 -0.0004492794
    ## X2           0.0020493098 -0.0004492794  0.0085404365

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

    ##               Estimate Std. Error    t value     Pr(>|t|)
    ## (Intercept)  0.9234582  0.1000242   9.232344 6.098617e-15
    ## X1           1.8661495  0.1037805  17.981689 1.193536e-32
    ## X2          -0.9780210  0.0930549 -10.510150 1.064895e-17

``` r
## Prediction for new covariate values
X_new <- data.frame(X1=rnorm(5), X2=rnorm(5))
predict(fit_lm, newdata = X_new)
```

    ##         1         2         3         4         5 
    ## 3.8167306 2.6142026 0.6305609 0.4226252 3.4082587

``` r
predict(fit_sada, newdata = X_new)
```

    ## [1] 3.743283 2.722711 0.553095 0.403401 3.365777
