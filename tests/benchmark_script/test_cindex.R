library(survival)
library(survcomp)
library(jsonlite)
library(riskRegression)
library(survC1)
library(survAUC)


# function to get metrics
get_cindex <- function(TR, TE, train.fit){

  # survival data object
  Surv.rsp <- survival::Surv(TR$time, TR$status)
  Surv.rsp.new <- survival::Surv(TE$time, TE$status)

  # survival probability KM estimate
  surv.prob <- unique(survfit(Surv(TE$time,TE$status)~1)$surv)
  surv.prob = surv.prob[-length(surv.prob)]

  # predictors train and test data
  lp <- predict(train.fit, newdata = TR)
  lpnew <- predict(train.fit, newdata = TE)

  # max time
  times <- sort(unique(TE$time[TE$status == 1]))
  last_time_event = F
  if(max(TE$time) %in% times){
    last_time_event = T
    times <- times[times < max(TE$time)]
  }
  max_times <- max(times)


  #
  # survival

  # survival: Harrell's C-index
  c_Harrell_survival <- survival::concordance(train.fit, newdata = TE)$concordance


  #
  # SurvAUC

  # SurvAUC: Uno's c index
  c_Uno_survAUC = survAUC::UnoC(Surv.rsp, Surv.rsp.new, lpnew, time = max_times)


  #
  # survC1

  # survC1: Uno's C-index
  c_Uno_survC1 = Est.Cval(cbind(TE$time, TE$status, lpnew), tau = max_times, nofit = TRUE)$Dhat

  #
  # survcomp:
  c_survcomp <- survcomp_concordance.index_modified(lpnew, TE$time, TE$status, method='noether')
  c_survcomp_conservative <- survcomp_concordance.index_modified(lpnew, TE$time, TE$status, method='conservative')

  # Harrell's c-index
  c_Harrell_survcomp = c_survcomp$c.index

  # Noether standard error
  c_se_noether_survcomp = c_survcomp$se
  ch_survcomp = c_survcomp$ch
  dh_survcomp = c_survcomp$dh
  weights_survcomp = c_survcomp$weights

  # conservative confidence interval (symmetric)
  c_lower_conservative_survcomp = c_survcomp_conservative$lower


  return(list(train_time = TR$time, train_status = TR$status,
              test_time = TE$time, test_status = TE$status,
              estimate = lpnew, times = times,
              surv.prob = surv.prob,
              c_Uno_survAUC = c_Uno_survAUC,
              c_Uno_survC1 = c_Uno_survC1,
              c_Harrell_survival = c_Harrell_survival,
              c_Harrell_survcomp = c_Harrell_survcomp,
              c_se_noether_survcomp = c_se_noether_survcomp,
              c_lower_conservative_survcomp = c_lower_conservative_survcomp,
              ch_survcomp = ch_survcomp,
              dh_survcomp = dh_survcomp,
              weights_survcomp = weights_survcomp

  ))


}

# print more outputs from function
add_survcomp_ouputs <- function(){
  survcomp_concordance.index_modified <<- survcomp::concordance.index

  print(body(survcomp_concordance.index_modified)[[46]])
  body(survcomp_concordance.index_modified)[[46]] <<- substitute(return(list(c.index = cindex,
                                                                             se = se, weights=weights,
                                                                             lower = lower, upper = upper,
                                                                             p.value = p, n = length(x2),
                                                                             data = data, comppairs = cscount,
                                                                             ch = ch,
                                                                             dh = dh)))
  print(body(survcomp_concordance.index_modified)[[46]])
}

add_survcomp_ouputs()

# create list to save cindexs
cindexs <- list()
i = 1


#
# lung dataset
#


lung <- survival::lung
lung$sex = lung$sex  == 1
lung$status = lung$status  == 2
lung$age = (lung$age - mean(lung$age)) / sd(lung$age)

midpoint = floor(nrow(lung) / 2)
mask = 1:nrow(lung) %in% 1:midpoint
TR <- lung[1:midpoint,]
TE <- lung[(midpoint + 1):nrow(lung),]
TR_complete <- TR[complete.cases(TR),]
TE_complete <- TE[complete.cases(TE),]


# one and two covariates and all covariates ignore missing values
train.fit_1 <- coxph(Surv(time, status) ~ age, data = TR, method = 'efron', x= T, y = T)
train.fit_2 <- coxph(Surv(time, status) ~ age + sex, data = TR, method = 'efron', x= T, y = T)
train.fit_3 <- coxph(Surv(time, status) ~ age + sex + ph.ecog + ph.karno +
                       pat.karno + meal.cal + wt.loss, data = TR_complete, method = 'efron', x= T, y = T)

# save metrics
cindexs[[i]] <- get_cindex(TR, TE, train.fit_1); i = i + 1
cindexs[[i]] <- get_cindex(TR, TE, train.fit_2); i = i + 1
cindexs[[i]] <- get_cindex(TR_complete, TE_complete, train.fit_3); i = i + 1


#
# gbsg dataset
#

gbsg <- survival::gbsg
gbsg$age = (gbsg$age - mean(gbsg$age)) / sd(gbsg$age)
gbsg$size = (gbsg$size - mean(gbsg$size)) / sd(gbsg$size)
gbsg$time = gbsg$rfstime

midpoint = floor(nrow(gbsg) / 2)
mask = 1:nrow(gbsg) %in% 1:midpoint
TR <- gbsg[1:midpoint,]
TE <- gbsg[(midpoint + 1):nrow(gbsg),]

# one and two covariates and all covariates
train.fit_1 <- coxph(Surv(rfstime, status) ~ age, data = TR, method = 'efron', x= T, y = T)
train.fit_2 <- coxph(Surv(rfstime, status) ~ age + size, data = TR, method = 'efron', x= T, y = T)
train.fit_3 <- coxph(Surv(rfstime, status) ~ age + size + grade +
                       nodes + pgr + er + hormon, data = TR, method = 'efron', x= T, y = T)

cindexs[[i]] <- get_cindex(TR, TE, train.fit_1); i = i + 1
cindexs[[i]] <- get_cindex(TR, TE, train.fit_2); i = i + 1
cindexs[[i]] <- get_cindex(TR, TE, train.fit_3); i = i + 1


#
# Save
#


write_json(cindexs, file.path("../benchmark_data/benchmark_cindex.json"))


#
# C-index
# Note: risksetROC::risksetAUC uses a smoothing methods to estimate AUC I/D that is integrated out to get the c-index
#       --> not compared to torchsurv
# Note: pec::cindex and Hmisc::rcorr.cens C-index does not consider ties in the calculation of their C-index (https://stackoverflow.com/questions/24784203/r-differences-between-cindex-function-from-pec-package-and-coxph)
#       --> not compared to torchsurv
# Note: SurvMetrics::Cindex does not consider ties but partial concordance metric (https://journal.r-project.org/articles/RJ-2023-009/RJ-2023-009.pdf)
#       --> not compared to torchsurv

