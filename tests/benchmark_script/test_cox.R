library(survival)
library(jsonlite)
library(data.table)

#
#
# create function to extract log hazard and log likelihood
#

get_log_likelihood <- function(formula, data, x, no_ties) {
  time <- data$time
  status <- data$status

  if (no_ties) {
    # if no ties, standard cox partial likelihood is used

    fit <- coxph(formula, data = data)
    log_hazard <- fit$coefficients %*% t(as.matrix(x, ncol = ncol(x)))
    log_likelihood <- as.numeric(logLik(fit))

    return(list(
      time = time,
      status = status,
      log_hazard = log_hazard,
      log_likelihood = log_likelihood,
      no_ties = no_ties
    ))
  } else {
    # with ties, use efrond and breslow method

    fit_efron <- coxph(formula, data = data, method = "efron")
    log_hazard_efron <- fit_efron$coefficients %*%
      t(as.matrix(x, ncol = ncol(x)))
    log_likelihood_efron <- as.numeric(logLik(fit_efron))

    fit_breslow <- coxph(formula, data = data, method = "breslow")
    log_hazard_breslow <- fit_breslow$coefficients %*%
      t(as.matrix(x, ncol = ncol(x)))
    log_likelihood_breslow <- as.numeric(logLik(fit_breslow))

    return(list(
      time = time,
      status = status,
      log_hazard_efron = log_hazard_efron,
      log_likelihood_efron = log_likelihood_efron,
      log_hazard_breslow = log_hazard_breslow,
      log_likelihood_breslow = log_likelihood_breslow,
      no_ties = no_ties
    ))
  }
}

# load lung dataset
lung <- survival::lung
lung$sex <- lung$sex == 1
lung$status <- lung$status == 2
lung$age <- (lung$age - mean(lung$age)) / sd(lung$age)

# load gbsg dataset
gbsg <- survival::gbsg
gbsg$age <- (gbsg$age - mean(gbsg$age)) / sd(gbsg$age)
gbsg$size <- (gbsg$size - mean(gbsg$size)) / sd(gbsg$size)
gbsg$time <- gbsg$rfstime


#
#
# 1. Covariates not-time varying
###########

#
#
# 1.1 No ties #########

# empty list to save log likelihoods
log_likelihoods <- list()
i <- 1

#
# lung dataset

# lung dataset without ties
index_duplicated <- which(duplicated(lung$time))
lung_unique <- lung[-index_duplicated, ]

# ignore missing values
lung_complete <- lung[complete.cases(lung), ]
lung_unique_complete <- lung_unique[complete.cases(lung_unique), ]

# log likelihoods
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age,
  data = lung_unique,
  x = lung_unique[, c("age")],
  no_ties = TRUE
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age + sex,
  data = lung_unique,
  x = lung_unique[, c("age", "sex")],
  no_ties = TRUE
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~
    age + sex + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss,
  data = lung_unique_complete,
  x = lung_unique_complete[, c(
    "age",
    "sex",
    "ph.ecog",
    "ph.karno",
    "pat.karno",
    "meal.cal",
    "wt.loss"
  )],
  no_ties = TRUE
)
i <- i + 1


#
# gbsg dataset

# gbsg dataset without ties
index_duplicated <- which(duplicated(gbsg$time))
gbsg_unique <- gbsg[-index_duplicated, ]

# log likelihood
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age,
  data = gbsg_unique,
  x = gbsg_unique[, c("age")],
  no_ties = TRUE
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(rfstime, status) ~ age + size,
  data = gbsg_unique,
  x = gbsg_unique[, c("age", "size")],
  no_ties = TRUE
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(rfstime, status) ~
    age + size + grade + nodes + pgr + er + hormon,
  data = gbsg_unique,
  x = gbsg_unique[, c("age", "size", "grade", "nodes", "pgr", "er", "hormon")],
  no_ties = TRUE
)
i <- i + 1

#
# Save
write_json(log_likelihoods, file.path("../benchmark_data/benchmark_cox_without_ties.json"))



#
#
# 1.2 With ties #########

# empty list to save log likelihoods
log_likelihoods <- list()
i <- 1

#
# lung dataset

# log likelihoods
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age,
  data = lung,
  x = lung[, c("age")],
  no_ties = FALSE
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age + sex,
  data = lung,
  x = lung[, c("age", "sex")],
  no_ties = FALSE
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~
    age + sex + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss,
  data = lung_complete,
  x = lung_complete[, c(
    "age",
    "sex",
    "ph.ecog",
    "ph.karno",
    "pat.karno",
    "meal.cal",
    "wt.loss"
  )],
  no_ties = FALSE
)
i <- i + 1


#
# gbsg

# log likehoods
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age,
  data = gbsg,
  x = gbsg[, c("age")],
  no_ties = FALSE
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(rfstime, status) ~ age + size,
  data = gbsg,
  x = gbsg[, c("age", "size")],
  no_ties = FALSE
)
i <- i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(rfstime, status) ~
    age + size + grade + nodes + pgr + er + hormon,
  data = gbsg,
  x = gbsg[, c("age", "size", "grade", "nodes", "pgr", "er", "hormon")],
  no_ties = FALSE
)
i <- i + 1

#
# Save
write_json(log_likelihoods, file.path("../benchmark_data/benchmark_cox_with_ties.json"))

#
#
# 2. Time-varying covariates
###########

#
#
# 2.1 Without ties #########


# create heart without ties
heart_subset <- as.data.table(heart[!heart$id %in% c(
  81, 103, 88, 101, 100, 58, 77, 97,
  99, 79, 30, 31, 95, 94, 20, 93, 92,
  21, 86, 70, 74, 85, 84, 19, 66, 83,
  80, 76, 43, 46, 61, 75, 73, 69, 68, 67,
  65, 23, 42, 54, 60, 53, 50, 17, 49, 47,
  15, 45, 44, 41, 22, 35, 33, 32
), ])

# prepare covariates
heart_subset[, max_time := max(stop), by = "id"]
heart_max_time <- heart_subset[stop == max_time]
stopifnot(nrow(heart_max_time) == length(unique(heart_subset$id)))
cov <- array(dim = c(length(heart_max_time$id), 3, length(heart_max_time$max_time)))
for (j in 1:length(heart_max_time$max_time)) {
  for (i in 1:length(heart_max_time$id)) {
    t <- heart_max_time$max_time[j]
    indiv <- heart_max_time$id[i]
    if (heart_max_time[id == indiv, max_time] < t) {
      cov[i, , j] <- c(0, 0, 0)
    } else {
      sub <- heart_subset[id == indiv & start < t & stop >= t]
      stopifnot(nrow(sub) == 1)
      cov[i, , j] <- c(sub$age, sub$surgery, sub$transplant)
    }
  }
}

# fit
fit <- coxph(Surv(start, stop, event) ~ age + surgery + transplant, data = heart_subset)
summary(fit)
log_likelihood <- as.numeric(logLik(fit))

# find log hazards
log_hazard <- array(dim = c(length(heart_max_time$id), length(heart_max_time$max_time)))
for (j in 1:length(heart_max_time$max_time)) {
  for (i in 1:length(heart_max_time$id)) {
    x <- cov[i, , j]
    log_hazard[i, j] <- sum(fit$coefficients * x)
  }
}

# save
log_likelihoods <- list(
  time = heart_max_time$max_time,
  status = heart_max_time$event,
  log_hazard = log_hazard,
  log_likelihood = log_likelihood,
  no_ties = TRUE
)

write_json(log_likelihoods, file.path("../benchmark_data/benchmark_extended_cox_without_ties.json"))





write_json(log_likelihoods, file.path("/Users/Monod/git/torchsurv/tests/benchmark_data/benchmark_extended_cox_without_ties.json"),
  digits = 16, # full double precision
  pretty = FALSE
)

#
# write_json(log_likelihoods, file.path("/Users/Monod/git/torchsurv/tests/benchmark_data/benchmark_cox.json"))

#
# # Ensure variables are numeric
# heart_subset[, surgery := as.numeric(surgery)]
# heart_subset[, transplant := as.numeric(transplant)]
#
# # Extract model matrix (automatically handles factors etc.)
# X <- model.matrix(~ age + surgery + transplant, data = heart_subset)[, -1]  # drop intercept
#
# # Coefficients from coxph
# beta <- coef(fit)
#
# # Linear predictor
# eta <- as.numeric(X %*% beta)
#
# # Event times
# event_times <- heart_subset$stop[heart_subset$event == 1]
#
# # Initialize log-likelihood
# loglik_manual <- 0
#
# # Loop over event times (Breslow version)
# for (t in event_times) {
#   at_risk <- (heart_subset$start < t) & (heart_subset$stop >= t)
#   event_idx <- (heart_subset$stop == t) & (heart_subset$event == 1)
#
#   loglik_manual <- loglik_manual +
#     sum(eta[event_idx]) -
#     log(sum(exp(eta[at_risk])))
# }
#
# loglik_manual
#
