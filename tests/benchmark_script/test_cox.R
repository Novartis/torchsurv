library(survival)
library(jsonlite)

# create function to extract log hazard and log likelihood
get_log_likelihood <- function(formula, data, x, no_ties){
  
  time = data$time
  status = data$status
  
  if(no_ties){
    # if no ties, standard cox partial likelihood is used
    
    fit <- coxph(formula, data = data)
    log_hazard <- fit$coefficients %*% t(as.matrix(x, ncol = ncol(x)))
    log_likelihood <- as.numeric(logLik(fit))
    
    return(list(time = time, status = status, 
                log_hazard = log_hazard, 
                log_likelihood = log_likelihood, 
                no_ties = no_ties))
  }else{
    # with ties, use efrond and breslow method
    
    fit_efron <- coxph(formula, data = data, method = 'efron')
    log_hazard_efron <- fit_efron$coefficients %*% t(as.matrix(x, ncol = ncol(x)))
    log_likelihood_efron <- as.numeric(logLik(fit_efron))
    
    fit_breslow <- coxph(formula, data = data, method = 'breslow')
    log_hazard_breslow <- fit_breslow$coefficients %*% t(as.matrix(x, ncol = ncol(x)))
    log_likelihood_breslow <- as.numeric(logLik(fit_breslow))
    
    return(list(time = time, status = status, 
                log_hazard_efron = log_hazard_efron, 
                log_likelihood_efron = log_likelihood_efron,
                log_hazard_breslow = log_hazard_breslow,
                log_likelihood_breslow = log_likelihood_breslow,
                no_ties = no_ties))
  }
  
}

#
# empty list to save log likelihoods
log_likelihoods <- list()
i = 1


#
# lung dataset
#

# load lung dataset
lung <- survival::lung
lung$sex = lung$sex  == 1
lung$status = lung$status  == 2
lung$age = (lung$age - mean(lung$age)) / sd(lung$age)

# lung dataset without ties
index_duplicated <- which(duplicated(lung$time))
lung_unique <- lung[-index_duplicated, ]

# ignore missing values
lung_complete <- lung[complete.cases(lung),]
lung_unique_complete <- lung_unique[complete.cases(lung_unique),]

# Cox: no ties
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age, 
  data = lung_unique, 
  x = lung_unique[, c('age')], 
  no_ties = TRUE); i = i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age + sex,
  data = lung_unique, 
  x = lung_unique[, c('age', 'sex')], 
  no_ties = TRUE); i = i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age + sex + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss,
  data = lung_unique_complete, 
  x = lung_unique_complete[, c('age', 'sex', 'ph.ecog', 'ph.karno', 'pat.karno', 'meal.cal', 'wt.loss')], 
  no_ties = TRUE); i = i + 1

# Cox with ties
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age, 
  data = lung, 
  x = lung[, c('age')], 
  no_ties = FALSE); i = i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age + sex,
  data = lung, 
  x = lung[, c('age', 'sex')], 
  no_ties = FALSE); i = i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age + sex + ph.ecog + ph.karno + pat.karno + meal.cal + wt.loss,
  data = lung_complete, 
  x = lung_complete[, c('age', 'sex', 'ph.ecog', 'ph.karno', 'pat.karno', 'meal.cal', 'wt.loss')], 
  no_ties = FALSE); i = i + 1


#
# gbsg dataset 
#

gbsg <- survival::gbsg
gbsg$age = (gbsg$age - mean(gbsg$age)) / sd(gbsg$age)
gbsg$size = (gbsg$size - mean(gbsg$size)) / sd(gbsg$size)
gbsg$time = gbsg$rfstime

# lung dataset without ties
index_duplicated <- which(duplicated(gbsg$time))
gbsg_unique <- gbsg[-index_duplicated, ]

# Cox: no ties
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age, 
  data = gbsg_unique, 
  x = gbsg_unique[, c('age')], 
  no_ties = TRUE); i = i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(rfstime, status) ~ age + size,
  data = gbsg_unique, 
  x = gbsg_unique[, c('age', 'size')], 
  no_ties = TRUE); i = i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(rfstime, status) ~ age + size + grade + nodes + pgr + er + hormon,
  data = gbsg_unique, 
  x = gbsg_unique[, c('age', 'size', 'grade', 'nodes', 'pgr', 'er', 'hormon')], 
  no_ties = TRUE); i = i + 1

# Cox with ties
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(time, status) ~ age, 
  data = gbsg, 
  x = gbsg[, c('age')], 
  no_ties = FALSE); i = i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(rfstime, status) ~ age + size,
  data = gbsg, 
  x = gbsg[, c('age', 'size')], 
  no_ties = FALSE); i = i + 1
log_likelihoods[[i]] <- get_log_likelihood(
  formula = Surv(rfstime, status) ~ age + size + grade + nodes + pgr + er + hormon,
  data = gbsg, 
  x = gbsg[, c('age', 'size', 'grade', 'nodes', 'pgr', 'er', 'hormon')], 
  no_ties = FALSE); i = i + 1


#
# Save
write_json(log_likelihoods, file.path("../benchmark_data/benchmark_cox.json"))

