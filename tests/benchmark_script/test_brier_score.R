library(survival)
library(survAUC)
library(SurvMetrics)


# function to get brier score
get_brier_score <- function(TR, TE, train.fit) {
  # survival data object
  Surv.rsp <- survival::Surv(TR$time, TR$status)
  Surv.rsp.new <- survival::Surv(TE$time, TE$status)

  # predictors train and test data
  lp <- predict(train.fit, newdata = TR)
  lpnew <- predict(train.fit, newdata = TE)

  # evaluation time
  times <- sort(unique(TE$time[TE$status == 1]))
  last_time_event <- F
  if (max(TE$time) %in% times) {
    last_time_event <- T
    times <- times[times < max(TE$time)]
  }

  # estimate of survival function
  estimate <- survival_function_weibull(train.fit, times, TE)

  #
  # SurvMetrics

  brier_score_survMetrics <- vector(mode = "numeric", length = length(times))
  for (i in seq_along(times)) {
    brier_score_survMetrics[i] <- Brier_SurvMetrics_DEBUG(
      Surv.rsp.new,
      estimate[, i],
      times[i]
    )
  }
  ibrier_score_survMetrics <- IBS_SurvMetrics_DEBUG(
    Surv.rsp.new,
    estimate,
    IBSrange = times
  )

  #
  # SurvAUC

  outputs_survAUC <- predErr(
    Surv.rsp,
    Surv.rsp.new,
    lp,
    lpnew,
    times,
    type = "brier",
    int.type = "unweighted"
  )
  brier_score_survAUC <- outputs_survAUC$error
  ibrier_score_survAUC <- outputs_survAUC$ierror

  return(list(
    train_time = TR$time,
    train_status = TR$status,
    test_time = TE$time,
    test_status = TE$status,
    estimate = estimate,
    times = times,
    brier_score_survMetrics = brier_score_survMetrics,
    ibrier_score_survMetrics = ibrier_score_survMetrics,
    brier_score_survAUC = brier_score_survAUC,
    ibrier_score_survAUC = ibrier_score_survAUC
  ))
}

# bugfix survMetrics brier score
bugfix_TimeROC_compute_iid_decomposition <- function() {
  Brier_SurvMetrics_DEBUG <<- SurvMetrics::Brier
  IBS_SurvMetrics_DEBUG <<- SurvMetrics::IBS

  # cases should be t <= t_star instead of t < t_star
  print(body(Brier_SurvMetrics_DEBUG)[[21]][[4]][[2]][[2]])
  body(Brier_SurvMetrics_DEBUG)[[21]][[4]][[2]][[2]] <<- substitute(
    time[i] <= t_star & (status[i] == 1)
  )
  print(body(Brier_SurvMetrics_DEBUG)[[21]][[4]][[2]][[2]])

  # controls should be t > t_star instead of t >= t_star
  print(body(Brier_SurvMetrics_DEBUG)[[21]][[4]][[3]][[2]])
  body(Brier_SurvMetrics_DEBUG)[[21]][[4]][[3]][[2]] <<- substitute(
    time[i] > t_star
  )
  print(body(Brier_SurvMetrics_DEBUG)[[21]][[4]][[3]][[2]])

  # use the Brier debug version in IBS
  print(body(IBS_SurvMetrics_DEBUG)[[14]][[3]][[2]])
  body(IBS_SurvMetrics_DEBUG)[[14]][[3]][[2]] <<- substitute(
    bs <- Brier_SurvMetrics_DEBUG(object, sp_matrix, IBSrange)
  )
  print(body(IBS_SurvMetrics_DEBUG)[[14]][[3]][[2]])

  print(body(IBS_SurvMetrics_DEBUG)[[14]][[4]][[4]][[3]][[4]][[4]][[4]])
  body(IBS_SurvMetrics_DEBUG)[[14]][[4]][[4]][[3]][[4]][[4]][[
    4
  ]] <<- substitute(
    t_brier[i] <- Brier_SurvMetrics_DEBUG(object, pre_sp, t_star)
  )
  print(body(IBS_SurvMetrics_DEBUG)[[14]][[4]][[4]][[3]][[4]][[4]][[4]])

  print(body(IBS_SurvMetrics_DEBUG)[[14]][[4]][[4]][[4]][[6]][[4]][[4]])
  body(IBS_SurvMetrics_DEBUG)[[14]][[4]][[4]][[4]][[6]][[4]][[
    4
  ]] <<- substitute(
    t_brier[i] <- Brier_SurvMetrics_DEBUG(object, pre_sp, t_star)
  )
  print(body(IBS_SurvMetrics_DEBUG)[[14]][[4]][[4]][[4]][[6]][[4]][[4]])
}
bugfix_TimeROC_compute_iid_decomposition()

# survival function weibull AFT model
survival_function_weibull <- function(fit, times, new_data) {
  survreg.lp <- predict(fit, type = "lp", newdata = new_data)
  survreg.scale <- fit$scale
  shape <- 1 / survreg.scale
  scale <- exp(survreg.lp)
  s <- matrix(ncol = length(times), nrow = length(scale), 0)
  for (i in seq_along(times)) {
    t <- times[i]
    s[, i] <- 1 - pweibull(t, shape = shape, scale = scale)
  }

  return(s)
}

# create list to save brier sores
brier_scores <- list()
i <- 1


#
# lung dataset
#

lung <- survival::lung
lung$sex <- lung$sex == 1
lung$status <- lung$status == 2
lung$age <- (lung$age - mean(lung$age)) / sd(lung$age)

midpoint <- floor(nrow(lung) / 2)
mask <- 1:nrow(lung) %in% 1:midpoint
TR <- lung[1:midpoint, ]
TE <- lung[(midpoint + 1):nrow(lung), ]
TR_complete <- TR[complete.cases(TR), ]
TE_complete <- TE[complete.cases(TE), ]


# one and two covariates and all covariates ignore missing values
train.fit_1 <- survreg(
  Surv(time, status) ~ age,
  data = TR,
  dist = "weibull",
  scale = 1,
  x = T,
  y = T
)
train.fit_2 <- survreg(
  Surv(time, status) ~ age + sex,
  data = TR,
  dist = "weibull",
  scale = 1,
  x = T,
  y = T
)
train.fit_3 <- survreg(
  Surv(time, status) ~
    age + sex + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss,
  data = TR_complete,
  dist = "weibull",
  scale = 1,
  x = T,
  y = T
)

# save metrics
brier_scores[[i]] <- get_brier_score(TR, TE, train.fit_1)
i <- i + 1
brier_scores[[i]] <- get_brier_score(TR, TE, train.fit_2)
i <- i + 1
brier_scores[[i]] <- get_brier_score(TR_complete, TE_complete, train.fit_3)
i <- i + 1


#
# gbsg dataset
#

gbsg <- survival::gbsg
gbsg$age <- (gbsg$age - mean(gbsg$age)) / sd(gbsg$age)
gbsg$size <- (gbsg$size - mean(gbsg$size)) / sd(gbsg$size)
gbsg$time <- gbsg$rfstime

midpoint <- floor(nrow(gbsg) / 2)
mask <- 1:nrow(gbsg) %in% 1:midpoint
TR <- gbsg[1:midpoint, ]
TE <- gbsg[(midpoint + 1):nrow(gbsg), ]

# one and two covariates and all covariates
train.fit_1 <- survreg(
  Surv(time, status) ~ age,
  data = TR,
  dist = "weibull",
  scale = 1,
  x = T,
  y = T
)
train.fit_2 <- survreg(
  Surv(time, status) ~ age + size,
  data = TR,
  dist = "weibull",
  scale = 1,
  x = T,
  y = T
)
train.fit_3 <- survreg(
  Surv(time, status) ~ age + size + grade + nodes + pgr + er + hormon,
  data = TR,
  dist = "weibull",
  scale = 1,
  x = T,
  y = T
)

brier_scores[[i]] <- get_brier_score(TR, TE, train.fit_1)
i <- i + 1
brier_scores[[i]] <- get_brier_score(TR, TE, train.fit_2)
i <- i + 1
brier_scores[[i]] <- get_brier_score(TR, TE, train.fit_3)
i <- i + 1


#
# Save
#

write_json(
  brier_scores,
  file.path("../benchmark_data/benchmark_brier_score.json")
)
