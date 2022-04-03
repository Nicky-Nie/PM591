Assignment 4
================
NickyNie
4/2/2022

### Conceptual

The goal of this exercise is to illustrate the grouping properties of
Ridge and Elastic-net compared to Lasso Compute the full path for of
solutions the Lasso, Ridge and elastic net on the data generated below
using `glmnet` to generate a full path of solutions:

``` r
library(glmnet)
```

    ## Warning: ³Ì¼­°ü'glmnet'ÊÇÓÃR°æ±¾4.1.3 À´½¨ÔìµÄ

    ## 载入需要的程辑包：Matrix

    ## Loaded glmnet 4.1-3

``` r
# Simulates Features
n = 50
x1 = rnorm(n)
x2 = x1
x3 = rnorm(n)
X = cbind(x1, x2, x3)

# Simulates error/noise
e = rnorm(n, sd = 0.1)

#Simulates outcome y
y = 1 + 2*x1 + e
```

``` r
lambda_grid = 10^seq(2,-2,length=50)
model_ridge = glmnet(X, y, family='gaussian', alpha=0, standardize=TRUE, lambda=lambda_grid)
model_ridge_coefs = coef(model_ridge)
round(model_ridge_coefs[, 43:50], 2)
```

    ## 4 x 8 sparse Matrix of class "dgCMatrix"
    ##              s42  s43  s44  s45  s46  s47  s48  s49
    ## (Intercept) 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
    ## x1          1.00 1.00 1.00 1.01 1.01 1.01 1.01 1.01
    ## x2          0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98
    ## x3          0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02

``` r
model_lasso = glmnet(X, y, family = "gaussian", alpha = 1)
model_lasso_coefs = coef(model_lasso)
round(model_lasso_coefs[, 43:50], 2)
```

    ## 4 x 8 sparse Matrix of class "dgCMatrix"
    ##              s42  s43  s44  s45  s46  s47  s48  s49
    ## (Intercept) 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
    ## x1          1.96 1.96 1.96 1.97 1.97 1.97 1.97 1.97
    ## x2          0.00 0.00 .    0.00 .    0.00 .    .   
    ## x3          .    .    .    .    .    .    .    .

``` r
model_enet = glmnet(X, y, family = "gaussian", alpha = 0.5)
model_enet_coefs = coef(model_enet)
round(model_enet_coefs[, 43:50], 2)
```

    ## 4 x 8 sparse Matrix of class "dgCMatrix"
    ##              s42  s43  s44  s45  s46  s47  s48  s49
    ## (Intercept) 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00
    ## x1          0.98 0.99 0.99 0.99 0.99 1.00 1.00 1.00
    ## x2          0.95 0.96 0.96 0.96 0.96 0.96 0.96 0.96
    ## x3          .    .    .    .    .    .    .    .

Comment on your results regarding grouping effects of the estimated
coefficients.

Ridge regression use all features to represent the group.

The outcome y is highly related with x1, so Lasso tends to select one
feature(x1) to represent the group.

But for elastic net regression, it tends to select the entire group and
‘spreads’ the effect assigning similarly estimated coefficients (x1 and
x2) across the features in the group.

### Analysis

#### Exercise 1

You will build a model to predict psa levels using PCA linear regression
using the PSA prostate data

1.  Load the mlr3 library and the prostate data

``` r
library(mlr3)
library(mlr3verse)

# read in data
prostate <- read.csv("prostate.csv")
```

2.  Specify the regression task and the base linear regression learner.
    Note: we will not split the data into training and testing because
    of the modest sample size. Explain whether this invalidates any
    prediction model we develop and why in practice we always want a
    test set.

``` r
# create PSA task
psa.tsk <- as_task_regr(prostate, target = "lpsa", id = "PSA Prediction") 

# create basic linear regression learner
lm.lrn <- lrn("regr.lm")
```

It does not invalidate the PCA since PCA is unsupervised, but it may
invalidate linear regression model since no test set will result in an
overfitting problem, which can not be generalized to other datasets in
practice.

3.  Create a new learner adding a PCA preprocessing step to the base
    learner. In `mlr3` parlance this is called a pipeline operator,
    `%>>%`. This becomes a new ‘composite’ learner that can be used just
    like any of the standard learners we used before. In particular if
    K-fold CV is used, both the PCA and the linear regression will be
    used for training on each set of K-1 folds and prediction on the
    K-th fold.

``` r
library(mlr3pipelines)

# create PCA step
pca <- po("pca")

# combines linear regression and PCA into a single learner
pca_lm.lrn <- pca %>>% lm.lrn
```

4.  Rather than fixing it as in the lecture, we will treat the number of
    principal components `pca.rank.` as a tuning parameter. Specify
    `pca.rank.` as a an integer tuning parameter ranging from 1 to the
    number of features in the PSA data

``` r
ps <- ps(pca.rank. = p_int(lower = 1, length(psa.tsk$feature_names)))
```

22. Create a control object for hyperparameter tuning with grid search.

``` r
ctrl <- tnr("grid_search", resolution = 8) 
# resolution is the number of points in the grid of values for the tuning parameter. Since there are 8 possible PCs we want resolution = 8 
```

6.  Perform the tuning

``` r
set.seed(202)

# resampling method
cv5 <- rsmp("cv", folds = 5)

# create the autotuner
pca_lm.lrn = AutoTuner$new(
  learner = pca_lm.lrn,
  resampling = cv5,
  measure = msr("regr.rsq"),
  search_space = ps,
  terminator = trm("evals", n_evals = 10), # stop after 10 iterations
  tuner = ctrl
)

# complete the tuning
lgr::get_logger("mlr3")$set_threshold("warn")
pca_lm.lrn$train(psa.tsk)
```

    ## INFO  [16:28:57.783] [bbotk] Starting to optimize 1 parameter(s) with '<TunerGridSearch>' and '<TerminatorEvals> [n_evals=10, k=0]' 
    ## INFO  [16:28:57.805] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:28:58.474] [bbotk] Result of batch 1: 
    ## INFO  [16:28:58.476] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:28:58.476] [bbotk]          8 0.5226138        0      0             0.52 
    ## INFO  [16:28:58.476] [bbotk]                                 uhash 
    ## INFO  [16:28:58.476] [bbotk]  b2b0fe64-6e3f-43b1-b83a-9159c38ab5e9 
    ## INFO  [16:28:58.477] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:28:58.997] [bbotk] Result of batch 2: 
    ## INFO  [16:28:58.998] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:28:58.998] [bbotk]          4 0.3597818        0      0             0.43 
    ## INFO  [16:28:58.998] [bbotk]                                 uhash 
    ## INFO  [16:28:58.998] [bbotk]  7e58d354-100d-49a6-b7b4-46e78ffab407 
    ## INFO  [16:28:59.000] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:28:59.620] [bbotk] Result of batch 3: 
    ## INFO  [16:28:59.622] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:28:59.622] [bbotk]          2 0.0765287        0      0             0.49 
    ## INFO  [16:28:59.622] [bbotk]                                 uhash 
    ## INFO  [16:28:59.622] [bbotk]  9d450c4a-da1e-4399-af51-0ed189e42c24 
    ## INFO  [16:28:59.624] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:00.132] [bbotk] Result of batch 4: 
    ## INFO  [16:29:00.133] [bbotk]  pca.rank.   regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:00.133] [bbotk]          1 0.08762075        0      0             0.39 
    ## INFO  [16:29:00.133] [bbotk]                                 uhash 
    ## INFO  [16:29:00.133] [bbotk]  fa179952-4987-42b8-b73b-568429f62020 
    ## INFO  [16:29:00.134] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:00.816] [bbotk] Result of batch 5: 
    ## INFO  [16:29:00.817] [bbotk]  pca.rank. regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:00.817] [bbotk]          3  0.19538        0      0             0.59 
    ## INFO  [16:29:00.817] [bbotk]                                 uhash 
    ## INFO  [16:29:00.817] [bbotk]  6a9fb332-fcaa-4c77-bcf7-2c9005c882e6 
    ## INFO  [16:29:00.818] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:01.300] [bbotk] Result of batch 6: 
    ## INFO  [16:29:01.302] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:01.302] [bbotk]          6 0.4559056        0      0             0.36 
    ## INFO  [16:29:01.302] [bbotk]                                 uhash 
    ## INFO  [16:29:01.302] [bbotk]  b6a86611-8597-41e5-a41f-16c637a930a9 
    ## INFO  [16:29:01.303] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:01.756] [bbotk] Result of batch 7: 
    ## INFO  [16:29:01.757] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:01.757] [bbotk]          7 0.4741076        0      0             0.38 
    ## INFO  [16:29:01.757] [bbotk]                                 uhash 
    ## INFO  [16:29:01.757] [bbotk]  962181c4-a836-477c-81af-54be75ca806b 
    ## INFO  [16:29:01.758] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:02.251] [bbotk] Result of batch 8: 
    ## INFO  [16:29:02.252] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:02.252] [bbotk]          5 0.4407682        0      0             0.39 
    ## INFO  [16:29:02.252] [bbotk]                                 uhash 
    ## INFO  [16:29:02.252] [bbotk]  4efda136-8a0d-40eb-a4e9-511360f3cf67 
    ## INFO  [16:29:02.257] [bbotk] Finished optimizing after 8 evaluation(s) 
    ## INFO  [16:29:02.258] [bbotk] Result: 
    ## INFO  [16:29:02.259] [bbotk]  pca.rank. learner_param_vals  x_domain  regr.rsq 
    ## INFO  [16:29:02.259] [bbotk]          8          <list[1]> <list[1]> 0.5226138

7.  How many principal components are selected? Does preprocessing by
    PCA help in this case?

8 principal components, preprocessing by PCA doesn’t help since
originally it has 8 PCs, no PCs are dropped.

8.  Use now benchmark to automate the comparison between PCA regression
    and standard linear regression on the prostate data

``` r
set.seed(101)
design = design = benchmark_grid(
  tasks = psa.tsk,
  learners = list(lm.lrn, pca_lm.lrn),
  resampling = rsmp("cv", folds = 5)
)
psa_benchmark = benchmark(design)
```

    ## INFO  [16:29:02.438] [bbotk] Starting to optimize 1 parameter(s) with '<TunerGridSearch>' and '<TerminatorEvals> [n_evals=10, k=0]' 
    ## INFO  [16:29:02.440] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:02.996] [bbotk] Result of batch 1: 
    ## INFO  [16:29:02.997] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:02.997] [bbotk]          8 0.3328886        0      0             0.45 
    ## INFO  [16:29:02.997] [bbotk]                                 uhash 
    ## INFO  [16:29:02.997] [bbotk]  f068c037-f885-44d6-b53d-16ae5e3d60d8 
    ## INFO  [16:29:02.998] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:03.563] [bbotk] Result of batch 2: 
    ## INFO  [16:29:03.566] [bbotk]  pca.rank.   regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:03.566] [bbotk]          2 -0.1672248        0      0             0.46 
    ## INFO  [16:29:03.566] [bbotk]                                 uhash 
    ## INFO  [16:29:03.566] [bbotk]  73be15d9-0197-4163-99f0-6b8720947f99 
    ## INFO  [16:29:03.567] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:04.114] [bbotk] Result of batch 3: 
    ## INFO  [16:29:04.117] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:04.117] [bbotk]          7 0.3366848        0      0             0.44 
    ## INFO  [16:29:04.117] [bbotk]                                 uhash 
    ## INFO  [16:29:04.117] [bbotk]  a25e250f-8549-405a-9652-69a5bf6710b8 
    ## INFO  [16:29:04.118] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:04.660] [bbotk] Result of batch 4: 
    ## INFO  [16:29:04.662] [bbotk]  pca.rank.    regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:04.662] [bbotk]          3 -0.03775769        0      0             0.42 
    ## INFO  [16:29:04.662] [bbotk]                                 uhash 
    ## INFO  [16:29:04.662] [bbotk]  cb21efb5-dd87-494a-8cfc-6cf981203988 
    ## INFO  [16:29:04.663] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:05.229] [bbotk] Result of batch 5: 
    ## INFO  [16:29:05.231] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:05.231] [bbotk]          5 0.3549156        0      0             0.51 
    ## INFO  [16:29:05.231] [bbotk]                                 uhash 
    ## INFO  [16:29:05.231] [bbotk]  6c62bc23-f217-4a97-a126-7a8dd3f85b42 
    ## INFO  [16:29:05.233] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:05.712] [bbotk] Result of batch 6: 
    ## INFO  [16:29:05.714] [bbotk]  pca.rank.    regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:05.714] [bbotk]          1 -0.07319588        0      0             0.35 
    ## INFO  [16:29:05.714] [bbotk]                                 uhash 
    ## INFO  [16:29:05.714] [bbotk]  398b8a87-e854-4af4-9f0c-86ad77fe8111 
    ## INFO  [16:29:05.715] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:06.216] [bbotk] Result of batch 7: 
    ## INFO  [16:29:06.218] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:06.218] [bbotk]          6 0.3412958        0      0             0.41 
    ## INFO  [16:29:06.218] [bbotk]                                 uhash 
    ## INFO  [16:29:06.218] [bbotk]  85324fe3-7177-41fb-b083-6f210a729b48 
    ## INFO  [16:29:06.219] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:06.698] [bbotk] Result of batch 8: 
    ## INFO  [16:29:06.699] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:06.699] [bbotk]          4 0.1977657        0      0             0.38 
    ## INFO  [16:29:06.699] [bbotk]                                 uhash 
    ## INFO  [16:29:06.699] [bbotk]  c8b72fc9-6039-4095-9f71-7f71cb4a24b2 
    ## INFO  [16:29:06.704] [bbotk] Finished optimizing after 8 evaluation(s) 
    ## INFO  [16:29:06.705] [bbotk] Result: 
    ## INFO  [16:29:06.706] [bbotk]  pca.rank. learner_param_vals  x_domain  regr.rsq 
    ## INFO  [16:29:06.706] [bbotk]          5          <list[1]> <list[1]> 0.3549156 
    ## INFO  [16:29:06.865] [bbotk] Starting to optimize 1 parameter(s) with '<TunerGridSearch>' and '<TerminatorEvals> [n_evals=10, k=0]' 
    ## INFO  [16:29:06.867] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:07.333] [bbotk] Result of batch 1: 
    ## INFO  [16:29:07.334] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:07.334] [bbotk]          1 0.1953748        0      0             0.41 
    ## INFO  [16:29:07.334] [bbotk]                                 uhash 
    ## INFO  [16:29:07.334] [bbotk]  bedb28dd-b738-4568-a6c8-5fa4561c8cd0 
    ## INFO  [16:29:07.335] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:07.819] [bbotk] Result of batch 2: 
    ## INFO  [16:29:07.821] [bbotk]  pca.rank.   regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:07.821] [bbotk]          3 0.08938841        0      0              0.4 
    ## INFO  [16:29:07.821] [bbotk]                                 uhash 
    ## INFO  [16:29:07.821] [bbotk]  75abfe80-7b75-47e6-a20c-256f2e559c2b 
    ## INFO  [16:29:07.824] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:08.302] [bbotk] Result of batch 3: 
    ## INFO  [16:29:08.304] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:08.304] [bbotk]          4 0.3808157        0      0              0.4 
    ## INFO  [16:29:08.304] [bbotk]                                 uhash 
    ## INFO  [16:29:08.304] [bbotk]  4734f202-7249-4e05-a3f0-4f5af3d77b4a 
    ## INFO  [16:29:08.305] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:08.784] [bbotk] Result of batch 4: 
    ## INFO  [16:29:08.786] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:08.786] [bbotk]          7 0.5368802        0      0             0.42 
    ## INFO  [16:29:08.786] [bbotk]                                 uhash 
    ## INFO  [16:29:08.786] [bbotk]  30832b86-0859-4a15-8e9e-1ffbd6f5add7 
    ## INFO  [16:29:08.787] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:09.307] [bbotk] Result of batch 5: 
    ## INFO  [16:29:09.309] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:09.309] [bbotk]          2 0.1277869        0      0             0.41 
    ## INFO  [16:29:09.309] [bbotk]                                 uhash 
    ## INFO  [16:29:09.309] [bbotk]  66ba2982-ecb1-44c0-b5ce-4bdb5ffa382b 
    ## INFO  [16:29:09.313] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:09.810] [bbotk] Result of batch 6: 
    ## INFO  [16:29:09.811] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:09.811] [bbotk]          5 0.4677768        0      0             0.41 
    ## INFO  [16:29:09.811] [bbotk]                                 uhash 
    ## INFO  [16:29:09.811] [bbotk]  13147c07-7ae2-4cc6-971c-f198d06aeeb3 
    ## INFO  [16:29:09.812] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:10.317] [bbotk] Result of batch 7: 
    ## INFO  [16:29:10.318] [bbotk]  pca.rank. regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:10.318] [bbotk]          8 0.519474        0      0             0.42 
    ## INFO  [16:29:10.318] [bbotk]                                 uhash 
    ## INFO  [16:29:10.318] [bbotk]  bf0869f5-e3b1-4088-8482-dc5d29e03771 
    ## INFO  [16:29:10.320] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:10.837] [bbotk] Result of batch 8: 
    ## INFO  [16:29:10.838] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:10.838] [bbotk]          6 0.4898779        0      0             0.44 
    ## INFO  [16:29:10.838] [bbotk]                                 uhash 
    ## INFO  [16:29:10.838] [bbotk]  13113010-e492-47a3-a0f5-c0c26393965a 
    ## INFO  [16:29:10.843] [bbotk] Finished optimizing after 8 evaluation(s) 
    ## INFO  [16:29:10.843] [bbotk] Result: 
    ## INFO  [16:29:10.844] [bbotk]  pca.rank. learner_param_vals  x_domain  regr.rsq 
    ## INFO  [16:29:10.844] [bbotk]          7          <list[1]> <list[1]> 0.5368802 
    ## INFO  [16:29:11.011] [bbotk] Starting to optimize 1 parameter(s) with '<TunerGridSearch>' and '<TerminatorEvals> [n_evals=10, k=0]' 
    ## INFO  [16:29:11.013] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:11.514] [bbotk] Result of batch 1: 
    ## INFO  [16:29:11.516] [bbotk]  pca.rank.   regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:11.516] [bbotk]          1 0.02039999        0      0             0.36 
    ## INFO  [16:29:11.516] [bbotk]                                 uhash 
    ## INFO  [16:29:11.516] [bbotk]  07ad9ae9-6e3f-434c-9eea-81e8f557b533 
    ## INFO  [16:29:11.517] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:12.023] [bbotk] Result of batch 2: 
    ## INFO  [16:29:12.025] [bbotk]  pca.rank.   regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:12.025] [bbotk]          2 0.01384339        0      0             0.41 
    ## INFO  [16:29:12.025] [bbotk]                                 uhash 
    ## INFO  [16:29:12.025] [bbotk]  4643fec2-645f-4c14-9be2-25ff261784e0 
    ## INFO  [16:29:12.026] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:12.549] [bbotk] Result of batch 3: 
    ## INFO  [16:29:12.551] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:12.551] [bbotk]          6 0.4791764        0      0             0.46 
    ## INFO  [16:29:12.551] [bbotk]                                 uhash 
    ## INFO  [16:29:12.551] [bbotk]  b76ceaeb-28b0-4159-9700-64b91629a703 
    ## INFO  [16:29:12.552] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:13.077] [bbotk] Result of batch 4: 
    ## INFO  [16:29:13.079] [bbotk]  pca.rank. regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:13.079] [bbotk]          5 0.477662        0      0             0.39 
    ## INFO  [16:29:13.079] [bbotk]                                 uhash 
    ## INFO  [16:29:13.079] [bbotk]  b7c45f26-7e21-4018-85ec-bbcc165082ce 
    ## INFO  [16:29:13.081] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:13.606] [bbotk] Result of batch 5: 
    ## INFO  [16:29:13.608] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:13.608] [bbotk]          8 0.5662808        0      0             0.42 
    ## INFO  [16:29:13.608] [bbotk]                                 uhash 
    ## INFO  [16:29:13.608] [bbotk]  9c794939-31e1-42f9-b65f-3cf59e7f61f8 
    ## INFO  [16:29:13.609] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:14.116] [bbotk] Result of batch 6: 
    ## INFO  [16:29:14.118] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:14.118] [bbotk]          4 0.3926575        0      0             0.42 
    ## INFO  [16:29:14.118] [bbotk]                                 uhash 
    ## INFO  [16:29:14.118] [bbotk]  096192dd-4529-4935-a0f7-244b9f7d8165 
    ## INFO  [16:29:14.120] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:14.635] [bbotk] Result of batch 7: 
    ## INFO  [16:29:14.644] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:14.644] [bbotk]          7 0.5074022        0      0             0.43 
    ## INFO  [16:29:14.644] [bbotk]                                 uhash 
    ## INFO  [16:29:14.644] [bbotk]  838df6dc-1e8e-4863-8f69-78d931ae512a 
    ## INFO  [16:29:14.645] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:15.367] [bbotk] Result of batch 8: 
    ## INFO  [16:29:15.368] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:15.368] [bbotk]          3 0.2018228        0      0             0.63 
    ## INFO  [16:29:15.368] [bbotk]                                 uhash 
    ## INFO  [16:29:15.368] [bbotk]  69cba6a3-9816-476e-9e74-042fd6298deb 
    ## INFO  [16:29:15.374] [bbotk] Finished optimizing after 8 evaluation(s) 
    ## INFO  [16:29:15.375] [bbotk] Result: 
    ## INFO  [16:29:15.376] [bbotk]  pca.rank. learner_param_vals  x_domain  regr.rsq 
    ## INFO  [16:29:15.376] [bbotk]          8          <list[1]> <list[1]> 0.5662808 
    ## INFO  [16:29:15.529] [bbotk] Starting to optimize 1 parameter(s) with '<TunerGridSearch>' and '<TerminatorEvals> [n_evals=10, k=0]' 
    ## INFO  [16:29:15.532] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:15.980] [bbotk] Result of batch 1: 
    ## INFO  [16:29:15.982] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:15.982] [bbotk]          7 0.5266281        0      0             0.36 
    ## INFO  [16:29:15.982] [bbotk]                                 uhash 
    ## INFO  [16:29:15.982] [bbotk]  a94d5f07-ff38-4294-a844-d8724153f8f3 
    ## INFO  [16:29:15.983] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:16.461] [bbotk] Result of batch 2: 
    ## INFO  [16:29:16.462] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:16.462] [bbotk]          3 0.2575396        0      0             0.42 
    ## INFO  [16:29:16.462] [bbotk]                                 uhash 
    ## INFO  [16:29:16.462] [bbotk]  2b00e1fc-33f5-4136-8c8c-d2f9aaad5283 
    ## INFO  [16:29:16.463] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:16.926] [bbotk] Result of batch 3: 
    ## INFO  [16:29:16.927] [bbotk]  pca.rank.   regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:16.927] [bbotk]          1 0.09814662        0      0             0.38 
    ## INFO  [16:29:16.927] [bbotk]                                 uhash 
    ## INFO  [16:29:16.927] [bbotk]  b7a312ce-54d4-437b-b908-ed2c6f420216 
    ## INFO  [16:29:16.928] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:17.403] [bbotk] Result of batch 4: 
    ## INFO  [16:29:17.405] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:17.405] [bbotk]          4 0.3511745        0      0              0.4 
    ## INFO  [16:29:17.405] [bbotk]                                 uhash 
    ## INFO  [16:29:17.405] [bbotk]  da919d93-9674-480f-a40d-83b35a328a8e 
    ## INFO  [16:29:17.407] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:17.861] [bbotk] Result of batch 5: 
    ## INFO  [16:29:17.864] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:17.864] [bbotk]          8 0.5664979        0      0             0.39 
    ## INFO  [16:29:17.864] [bbotk]                                 uhash 
    ## INFO  [16:29:17.864] [bbotk]  6a140d87-14e9-4d42-94ea-acf0bfa88ce7 
    ## INFO  [16:29:17.865] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:18.367] [bbotk] Result of batch 6: 
    ## INFO  [16:29:18.368] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:18.368] [bbotk]          6 0.4851677        0      0             0.41 
    ## INFO  [16:29:18.368] [bbotk]                                 uhash 
    ## INFO  [16:29:18.368] [bbotk]  13151583-d7f7-47fc-8f6d-da536a86de5f 
    ## INFO  [16:29:18.369] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:18.840] [bbotk] Result of batch 7: 
    ## INFO  [16:29:18.841] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:18.841] [bbotk]          5 0.4590813        0      0             0.36 
    ## INFO  [16:29:18.841] [bbotk]                                 uhash 
    ## INFO  [16:29:18.841] [bbotk]  7125c19b-29af-4052-bf8b-9b090034a7da 
    ## INFO  [16:29:18.843] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:19.303] [bbotk] Result of batch 8: 
    ## INFO  [16:29:19.305] [bbotk]  pca.rank.   regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:19.305] [bbotk]          2 0.08689883        0      0             0.36 
    ## INFO  [16:29:19.305] [bbotk]                                 uhash 
    ## INFO  [16:29:19.305] [bbotk]  e97201b4-eec9-4a4d-b8fb-9d3a220dc117 
    ## INFO  [16:29:19.310] [bbotk] Finished optimizing after 8 evaluation(s) 
    ## INFO  [16:29:19.310] [bbotk] Result: 
    ## INFO  [16:29:19.311] [bbotk]  pca.rank. learner_param_vals  x_domain  regr.rsq 
    ## INFO  [16:29:19.311] [bbotk]          8          <list[1]> <list[1]> 0.5664979 
    ## INFO  [16:29:19.486] [bbotk] Starting to optimize 1 parameter(s) with '<TunerGridSearch>' and '<TerminatorEvals> [n_evals=10, k=0]' 
    ## INFO  [16:29:19.488] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:20.023] [bbotk] Result of batch 1: 
    ## INFO  [16:29:20.024] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:20.024] [bbotk]          5 0.4582161        0      0             0.43 
    ## INFO  [16:29:20.024] [bbotk]                                 uhash 
    ## INFO  [16:29:20.024] [bbotk]  55d5f6d4-9e75-4a2e-a0ba-6dcca0629ed3 
    ## INFO  [16:29:20.026] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:20.596] [bbotk] Result of batch 2: 
    ## INFO  [16:29:20.598] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:20.598] [bbotk]          4 0.3703437        0      0             0.44 
    ## INFO  [16:29:20.598] [bbotk]                                 uhash 
    ## INFO  [16:29:20.598] [bbotk]  553855b0-bce1-4922-afa2-4c63454fb0b9 
    ## INFO  [16:29:20.599] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:21.241] [bbotk] Result of batch 3: 
    ## INFO  [16:29:21.242] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:21.242] [bbotk]          6 0.4537063        0      0             0.52 
    ## INFO  [16:29:21.242] [bbotk]                                 uhash 
    ## INFO  [16:29:21.242] [bbotk]  918a100e-5553-4fb5-ba1b-ba007b746b42 
    ## INFO  [16:29:21.243] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:21.892] [bbotk] Result of batch 4: 
    ## INFO  [16:29:21.895] [bbotk]  pca.rank.   regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:21.895] [bbotk]          3 0.05767081        0      0             0.54 
    ## INFO  [16:29:21.895] [bbotk]                                 uhash 
    ## INFO  [16:29:21.895] [bbotk]  d29685eb-7373-4087-b7ae-025f3e186d32 
    ## INFO  [16:29:21.897] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:22.539] [bbotk] Result of batch 5: 
    ## INFO  [16:29:22.541] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:22.541] [bbotk]          7 0.4842443        0      0             0.48 
    ## INFO  [16:29:22.541] [bbotk]                                 uhash 
    ## INFO  [16:29:22.541] [bbotk]  b8d06375-c37b-4fc1-b718-4ba36bf9e0d4 
    ## INFO  [16:29:22.542] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:23.148] [bbotk] Result of batch 6: 
    ## INFO  [16:29:23.149] [bbotk]  pca.rank.    regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:23.149] [bbotk]          1 -0.01925737        0      0             0.52 
    ## INFO  [16:29:23.149] [bbotk]                                 uhash 
    ## INFO  [16:29:23.149] [bbotk]  f264e906-4f70-4ae3-b3ec-329ed5b89aef 
    ## INFO  [16:29:23.151] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:23.767] [bbotk] Result of batch 7: 
    ## INFO  [16:29:23.768] [bbotk]  pca.rank.    regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:23.768] [bbotk]          2 -0.05988657        0      0             0.54 
    ## INFO  [16:29:23.768] [bbotk]                                 uhash 
    ## INFO  [16:29:23.768] [bbotk]  50b10e72-0c75-4422-a31a-75024b593c8c 
    ## INFO  [16:29:23.769] [bbotk] Evaluating 1 configuration(s) 
    ## INFO  [16:29:24.260] [bbotk] Result of batch 8: 
    ## INFO  [16:29:24.262] [bbotk]  pca.rank.  regr.rsq warnings errors runtime_learners 
    ## INFO  [16:29:24.262] [bbotk]          8 0.5222638        0      0              0.4 
    ## INFO  [16:29:24.262] [bbotk]                                 uhash 
    ## INFO  [16:29:24.262] [bbotk]  c3ef8b27-7ced-46a5-b9cc-f156f17c5417 
    ## INFO  [16:29:24.267] [bbotk] Finished optimizing after 8 evaluation(s) 
    ## INFO  [16:29:24.267] [bbotk] Result: 
    ## INFO  [16:29:24.268] [bbotk]  pca.rank. learner_param_vals  x_domain  regr.rsq 
    ## INFO  [16:29:24.268] [bbotk]          8          <list[1]> <list[1]> 0.5222638

``` r
psa_benchmark$aggregate(msr('regr.rsq'))
```

    ##    nr      resample_result        task_id        learner_id resampling_id iters
    ## 1:  1 <ResampleResult[22]> PSA Prediction           regr.lm            cv     5
    ## 2:  2 <ResampleResult[22]> PSA Prediction pca.regr.lm.tuned            cv     5
    ##     regr.rsq
    ## 1: 0.5551278
    ## 2: 0.5332081

Still no obvious improvement for including PCA preprocessing in the
prediction model.

#### Exercise 2

You will build a classifier to predict cancer specific death among
breast cancer patients within 5-year of diagnosis based on a subset of
1,000 gene expression features from the Metabric data using ridge, lasso
and elastic net logistic regression. (The metabric data contains close
to 30,000 gene expression features, here we use a subset to keep
computation times reasonable for this in class Lab. In the homework
version you will use the full feature set)

1.  Load the Metabric data

``` r
load('metabric.Rdata') 
```

2.  Check the dimension of the metabric dataframe using `dim` check the
    number of deaths using `table` on the binary outcome variable

``` r
# check dimensions
cat("Dataset Dimensions: \n"); dim(metabric)
```

    ## Dataset Dimensions:

    ## [1]  803 1001

``` r
# make sure to factor outcome variable
metabric$y <- factor(metabric$y, labels=c("survive", "die"))

# check number of deaths
cat("Number of deaths: \n"); table(metabric$y)
```

    ## Number of deaths:

    ## 
    ## survive     die 
    ##     657     146

3.  Create an appropriate mlr3 task

``` r
metabric.tsk <- as_task_classif(metabric, target = "y", id = "One-year Breast Cancer Mortality")
```

4.  Split the data into training (70%) and test (30%)

``` r
set.seed(123)

# specify resampling to have 70/30 training/testing split
holdout.desc <- rsmp("holdout", ratio = 0.7)

# instantiate split
holdout.desc$instantiate(metabric.tsk)

# extract training and testing sets
train <- holdout.desc$train_set(1)
test  <- holdout.desc$test_set(1)
```

22. Create lasso, ridge, and Elastic net learners using
    “classif.cv_glmnet” (Recall that by specifying `cv.glmnet` as the
    learner, k-fold (10-fold by default) will be automatically used to
    tune the lambda penalty parameter. This takes advantage of the fast
    implementation of cross-validation within the `glmnet` package
    rather than cross-validating using `mlr3` tools).

``` r
# LASSO
lasso.lrn <- lrn("classif.cv_glmnet", 
                  alpha = 1, 
                  type.measure = "auc") 

lasso.lrn$predict_type <- "prob"

# ridge
ridge.lrn <- lrn("classif.cv_glmnet",
                 alpha=0, standardize=TRUE, lambda=lambda_grid,
                 type.measure = "auc")
ridge.lrn$predict_type <- "prob"

#Elastic net
enet.lrn <- lrn("classif.cv_glmnet",
                  alpha = 0.5,
                type.measure = "auc")
enet.lrn$predict_type <- "prob"
```

6.  Train the models on the training data using CV with an appropriate
    performance measure (hint: you can check the available measures for
    your task using `listMeasures`). Extract the cross-validated measure
    of performance. Why is the CV measure of performance the relevant
    metric to compare models?

``` r
# LASSO
lasso.lrn$train(metabric.tsk, row_ids=train)
lasso.lrn$model
```

    ## 
    ## Call:  (if (cv) glmnet::cv.glmnet else glmnet::glmnet)(x = data, y = target,      type.measure = "auc", alpha = 1, family = "binomial") 
    ## 
    ## Measure: AUC 
    ## 
    ##      Lambda Index Measure      SE Nonzero
    ## min 0.07806    10  0.7612 0.02213       5
    ## 1se 0.09850     5  0.7414 0.02327       4

``` r
cat("LASSO cross-validated AUC:");max(lasso.lrn$model$cvm)
```

    ## LASSO cross-validated AUC:

    ## [1] 0.7611816

``` r
# Ridge
ridge.lrn$train(metabric.tsk, row_ids=train)
ridge.lrn$model
```

    ## 
    ## Call:  (if (cv) glmnet::cv.glmnet else glmnet::glmnet)(x = data, y = target,      lambda = c(100, 82.8642772854684, 68.66488450043, 56.898660290183,      47.1486636345739, 39.0693993705462, 32.3745754281764, 26.8269579527973,      22.229964825262, 18.4206996932672, 15.2641796717523, 12.648552168553,      10.4811313415469, 8.68511373751353, 7.19685673001152, 5.96362331659465,      4.94171336132384, 4.09491506238043, 3.39322177189533, 2.81176869797423,      2.32995181051537, 1.93069772888325, 1.59985871960606, 1.32571136559011,      1.09854114198756, 0.910298177991522, 0.754312006335462, 0.625055192527398,      0.517947467923121, 0.429193426012878, 0.355648030622313,      0.294705170255181, 0.244205309454865, 0.202358964772516,      0.167683293681101, 0.138949549437314, 0.115139539932645,      0.0954095476349994, 0.079060432109077, 0.0655128556859552,      0.0542867543932386, 0.0449843266896945, 0.0372759372031494,      0.0308884359647748, 0.0255954792269954, 0.0212095088792019,      0.0175751062485479, 0.0145634847750124, 0.0120679264063933,      0.01), type.measure = "auc", alpha = 0, standardize = TRUE,      family = "binomial") 
    ## 
    ## Measure: AUC 
    ## 
    ##     Lambda Index Measure      SE Nonzero
    ## min   3.39    19  0.7709 0.01550    1000
    ## 1se 100.00     1  0.7670 0.01176    1000

``` r
cat("Ridge cross-validated AUC:");max(ridge.lrn$model$cvm)
```

    ## Ridge cross-validated AUC:

    ## [1] 0.7709296

``` r
# Elastic net
enet.lrn$train(metabric.tsk, row_ids=train)
enet.lrn$model
```

    ## 
    ## Call:  (if (cv) glmnet::cv.glmnet else glmnet::glmnet)(x = data, y = target,      type.measure = "auc", alpha = 0.5, family = "binomial") 
    ## 
    ## Measure: AUC 
    ## 
    ##     Lambda Index Measure      SE Nonzero
    ## min 0.1076    18  0.7592 0.01783      15
    ## 1se 0.1795     7  0.7444 0.02148       5

``` r
cat("Elastic net cross-validated AUC:");max(enet.lrn$model$cvm)
```

    ## Elastic net cross-validated AUC:

    ## [1] 0.7592243

Because simple validation only use part of data for training and the
performance of model highly depends on how you split the data, cross
validation provide a more convincing aspect of how well the prediction
model can be generalized on other data and also compared to other models
considered.

7.  Which method performs best? What does this say about the likely
    nature of the true relationship between the expression features and
    the outcome?

Ridge regression performs best. The expression features are not highly
correlated with each other and all expression features will affect the
outcome.

8.  Report an ‘honest’ estimate of prediction performance, plot the ROC
    curve.

``` r
#Fill in the ...
preds <- ridge.lrn$predict(metabric.tsk)
autoplot(preds, type= 'roc')
```

![](Assignment4_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

9.  Re-train the best performing method on all the data (training and
    test). This is the final model you would use to predict death in new
    women just diagnosed and treated for breast cancer. Why is this ok
    and why is this better than simply using the model trained on just
    the training data?

``` r
#Fill in the ...
metabric_final = ridge.lrn$train(metabric.tsk)
preds_final <- ridge.lrn$predict(metabric.tsk)
head(preds_final$prob)
```

    ##        survive       die
    ## [1,] 0.8257469 0.1742531
    ## [2,] 0.8078574 0.1921426
    ## [3,] 0.8204938 0.1795062
    ## [4,] 0.8045521 0.1954479
    ## [5,] 0.8278031 0.1721969
    ## [6,] 0.8199722 0.1800278

``` r
hist(preds_final$prob[,1])
```

![](Assignment4_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` r
p0.5 = median(preds_final$prob[,1])  # median predicted probability
preds_final$set_threshold(p0.5) # change prediction cutoff 
preds_final$confusion
```

    ##          truth
    ## response  survive die
    ##   survive     384  18
    ##   die         273 128

``` r
autoplot(preds_final, type= 'roc')
```

![](Assignment4_files/figure-gfm/unnamed-chunk-20-2.png)<!-- -->

Because cross validation already make sure that this model is best
prediction model for generalized data among all the three models, and
more data for training will improve the prediction performance of the
model with lower bias.

24. The dataset `new_expression_profiles` contains the gene expression
    levels for 15 women newly diagnosed with breast cancer. Estimate
    their one-year survival probabilities using the selected model.

``` r
# read in data
new_expression_profiles <- read.csv("new_expression_profiles.csv", header=T)

# predict in new data
predict_new <- metabric_final$predict_newdata(new_expression_profiles)
predict_new$prob
```

    ##         survive       die
    ##  [1,] 0.8068330 0.1931670
    ##  [2,] 0.8089030 0.1910970
    ##  [3,] 0.8217332 0.1782668
    ##  [4,] 0.8163726 0.1836274
    ##  [5,] 0.8137075 0.1862925
    ##  [6,] 0.8128847 0.1871153
    ##  [7,] 0.8240510 0.1759490
    ##  [8,] 0.8190984 0.1809016
    ##  [9,] 0.8231543 0.1768457
    ## [10,] 0.8182366 0.1817634
    ## [11,] 0.8202304 0.1797696
    ## [12,] 0.8253507 0.1746493
    ## [13,] 0.8244871 0.1755129
    ## [14,] 0.8215123 0.1784877
    ## [15,] 0.8156138 0.1843862

``` r
predict_new$response
```

    ##  [1] survive survive survive survive survive survive survive survive survive
    ## [10] survive survive survive survive survive survive
    ## Levels: survive die

11. Redo the model comparison between lasso, ridge, elastic net using
    mlr3’s `benchmark` function rather than manually

``` r
set.seed(101)
design = design = benchmark_grid(
    tasks = metabric.tsk,
    learners = list(ridge.lrn, lasso.lrn, enet.lrn),
    resampling = rsmp("cv", folds = 5)
)
metabric_benchmark = benchmark(design)
metabric_benchmark$aggregate(msr('classif.auc'))
```

    ##    nr      resample_result                          task_id        learner_id
    ## 1:  1 <ResampleResult[22]> One-year Breast Cancer Mortality classif.cv_glmnet
    ## 2:  2 <ResampleResult[22]> One-year Breast Cancer Mortality classif.cv_glmnet
    ## 3:  3 <ResampleResult[22]> One-year Breast Cancer Mortality classif.cv_glmnet
    ##    resampling_id iters classif.auc
    ## 1:            cv     5   0.7534616
    ## 2:            cv     5   0.7340771
    ## 3:            cv     5   0.7302805
