library(survival)
library(jsonlite)

# create function to extract log hazard and log likelihood
get_log_likelihood <- function(formula, data, x) {
  time <- data$time
  status <- data$status

  fit <- survreg(formula, data = data, dist = "weibull", scale = 1) # this is the survreg output model
  log_shape <- fit$coefficients %*% t(as.matrix(x, ncol = ncol(x)))
  log_likelihood <- fit$loglik[length(fit$loglik)]

  return(list(
    time = time,
    status = status,
    log_shape = log_shape,
    log_likelihood = log_likelihood
  ))
}

# empty list to save log likelihoods
log_likelihoods <- list()
i <- 1


# load lung dataset
lung <- survival::lung
lung$sex <- lung$sex == 1
lung$status <- lung$status == 2
lung$age <- (lung$age - mean(lung$age)) / sd(lung$age)

# lung dataset without ties
index_duplicated <- which(duplicated(lung$time))

# ignore missing values
lung_complete <- lung[complete.cases(lung), ]

log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age,
  data = lung,
  x = data.frame(rep(1, nrow(lung)), lung[, c("age")])
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age + sex,
  data = lung,
  x = data.frame(rep(1, nrow(lung)), lung[, c("age", "sex")])
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~
    age + sex + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss,
  data = lung_complete,
  x = data.frame(
    rep(1, nrow(lung_complete)),
    lung_complete[, c(
      "age",
      "sex",
      "ph.ecog",
      "ph.karno",
      "pat.karno",
      "meal.cal",
      "wt.loss"
    )]
  )
)
i <- i + 1


#
# gbsg dataset
#

gbsg <- survival::gbsg
gbsg$age <- (gbsg$age - mean(gbsg$age)) / sd(gbsg$age)
gbsg$size <- (gbsg$size - mean(gbsg$size)) / sd(gbsg$size)
gbsg$time <- gbsg$rfstime

log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age,
  data = gbsg,
  x = data.frame(rep(1, nrow(gbsg)), gbsg[, c("age")])
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(rfstime, status) ~ age + size,
  data = gbsg,
  x = data.frame(rep(1, nrow(gbsg)), gbsg[, c("age", "size")])
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(rfstime, status) ~
    age + size + grade + nodes + pgr + er + hormon,
  data = gbsg,
  x = data.frame(
    rep(1, nrow(gbsg)),
    gbsg[, c("age", "size", "grade", "nodes", "pgr", "er", "hormon")]
  )
)
i <- i + 1


#
# Save
write_json(
  log_likelihoods,
  file.path("../benchmark_data/benchmark_weibull.json")
)
