library(survival)
library(jsonlite)
library(riskRegression)
library(timeROC)
library(survAUC)

# function to get metrics
get_auc <- function(TR, TE, train.fit){
  
  # survival data object
  Surv.rsp <- survival::Surv(TR$time, TR$status)
  Surv.rsp.new <- survival::Surv(TE$time, TE$status)
  
  # times to estimate auc
  times <- sort(unique(TE$time[TE$status == 1]))
  last_time_event = F
  if(max(TE$time) %in% times){
    last_time_event = T
    times <- times[times < max(TE$time)]
    
  }
  
  # survival probability KM estimate
  surv.prob <- unique(survfit(Surv(TE$time,TE$status)~1)$surv)
  surv.prob = surv.prob[-length(surv.prob)]
  
  # predictors train and test data
  lp <- predict(train.fit, newdata = TR)
  lpnew <- predict(train.fit, newdata = TE)
  
  
  #
  # SurvAUC
  
  # SurvAUC: AUC C/D with ipcw
  AUC_Uno <- survAUC::AUC.uno(Surv.rsp, Surv.rsp.new, lpnew, times)
  auc_cd_survAUC <- AUC_Uno$auc
  
  # SurvAUC: integral of C/D with ipcw
  iauc_cd_survAUC <- survAUC::IntAUC(AUC_Uno$auc, AUC_Uno$times, surv.prob, max(times), auc.type="cumulative")
  
  # SurvAUC: Integral of AUC I/D Soug and Zhou (2008) definition
  auc_SZ <- survAUC::AUC.sh(Surv.rsp, Surv.rsp.new, lp, lpnew, times, type="incident")
  auc_id_sz_survAUC <- auc_SZ$auc
  i_auc_id_sz_survAUC <- auc_SZ$iauc
  
  
  #
  # TimeROC
  # There is a bug when auc is evaluated at event time because case are defined 
  # as T < t instead of T <= t. So instead we samples new_time in between the 
  # event times.
  times_se <- quantile(times, probs=seq(0.2,0.8,0.1))
  
  # TimeROC: Uno's AUC C/D
  auc_cd_Uno_timeROC <- timeROC_DEBUG(T=TE$time,
                                      delta=TE$status,
                                      marker=lpnew,
                                      cause=1,
                                      weighting="marginal",
                                      times=times_se,
                                      iid=TRUE)$AUC
  
  # TimeROC: standard error of Uno's AUC C/D
  auc_cd_Uno_se_timeROC <- timeROC_DEBUG(T=TE$time,
                                         delta=TE$status,
                                         marker=lpnew,
                                         cause=1,
                                         weighting="marginal",
                                         times=times_se,
                                         iid=TRUE)$inference$vect_sd_1
  
  
  #
  # riskRegression
  
  # riskRegression: Uno's AUC C/D 
  auc_cd_Uno_riskRegression <- riskRegression::Score(list("model"=train.fit),
                                                     formula=Surv(time,status)~1,
                                                     data=TE,
                                                     cens.method = "ipcw",
                                                     metrics="auc",
                                                     times=times)$AUC$score$AUC
  
  # riskRegression: standard error of Uno's AUC C/D 
  auc_cd_Uno_se_riskRegression <- riskRegression::Score(list("model"=train.fit),
                                                        formula=Surv(time,status)~1,
                                                        data=TE,
                                                        cens.method = "ipcw",
                                                        metrics="auc",
                                                        times=times)$AUC$score$se
  
  return(list(train_time = TR$time, train_status = TR$status, 
              test_time = TE$time, test_status = TE$status,
              estimate = lpnew, times = times,
              surv.prob = surv.prob,
              auc_cd_survAUC = auc_cd_survAUC, iauc_cd_survAUC = iauc_cd_survAUC, 
              auc_id_sz_survAUC = auc_id_sz_survAUC, i_auc_id_sz_survAUC = i_auc_id_sz_survAUC,
              auc_cd_Uno_riskRegression = auc_cd_Uno_riskRegression,
              auc_cd_Uno_se_riskRegression = auc_cd_Uno_se_riskRegression,
              times_se = times_se,
              auc_cd_Uno_timeROC = auc_cd_Uno_timeROC,
              auc_cd_Uno_se_timeROC = auc_cd_Uno_se_timeROC
              
  ))
  
  
}

# bugfix timeROC function
bugfix_TimeROC_compute_iid_decomposition <- function(){
  timeROC_DEBUG <<-timeROC::timeROC
  compute_iid_decomposition_DEBUG <<- timeROC:::compute_iid_decomposition
  compute_iid_decomposition_survival_DEBUG <<- timeROC:::compute_iid_decomposition_survival
  
  print(body(timeROC_DEBUG)[[41]][[3]][[2]][[4]][[15]][[4]][[2]][[3]][[2]][[3]][[1]])
  body(timeROC_DEBUG)[[41]][[3]][[2]][[4]][[15]][[4]][[2]][[3]][[2]][[3]][[1]] <<- substitute(compute_iid_decomposition_DEBUG)
  print(body(timeROC_DEBUG)[[41]][[3]][[2]][[4]][[15]][[4]][[2]][[3]][[2]][[3]][[1]])
  
  print(body(compute_iid_decomposition_DEBUG)[[5]][[4]][[2]][[1]])
  body(compute_iid_decomposition_DEBUG)[[5]][[4]][[2]][[1]] <<- substitute(compute_iid_decomposition_survival_DEBUG)
  print(body(compute_iid_decomposition_DEBUG)[[5]][[4]][[2]][[1]])
  
  # wrong division by St to get ipsw
  print(body(compute_iid_decomposition_survival_DEBUG)[[28]])
  body(compute_iid_decomposition_survival_DEBUG)[[28]] <<- substitute(
    vect_Tisupt <- as.numeric(Mat_data[, c("T")] >= t)/St
  )
}


# bugfix timeROC function
bugfix_TimeROC_compute_iid_decomposition()

# create list to save aucs
aucs <- list()
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

# save aucs
aucs[[i]] <- get_auc(TR, TE, train.fit_1); i = i + 1
aucs[[i]] <- get_auc(TR, TE, train.fit_2); i = i + 1
aucs[[i]] <- get_auc(TR_complete, TE_complete, train.fit_3); i = i + 1


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

aucs[[i]] <- get_auc(TR, TE, train.fit_1); i = i + 1
aucs[[i]] <- get_auc(TR, TE, train.fit_2); i = i + 1
aucs[[i]] <- get_auc(TR, TE, train.fit_3); i = i + 1


#
# Save
#


write_json(aucs, file.path("../benchmark_data/benchmark_auc.json"))


#
# AUC I/D
# Note: risksetROC::risksetAUC uses a smoothing methods to estimate AUC I/D
#       --> not compared to torchsurv


