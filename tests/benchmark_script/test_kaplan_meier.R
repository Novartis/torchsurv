library(survival)

get_kaplan_meier <- function(data){
  
  surv_fit <- survfit(Surv(time, status) ~ 1, data=data)
  times = surv_fit$time
  surv_prob_survival <- surv_fit$surv

  return(list(time=data$time, status=data$status, 
              times = times, 
              surv_prob_survival=surv_prob_survival))
}

# create list to save aucs
kaplan_meiers <- list()
i = 1

#
# lung dataset 

lung <- survival::lung
lung$sex = lung$sex  == 1
lung$status = lung$status  == 2
lung$age = (lung$age - mean(lung$age)) / sd(lung$age)

kaplan_meiers[[1]] <- get_kaplan_meier(lung)

#
# gbsg dataset 

gbsg <- survival::gbsg
gbsg$age = (gbsg$age - mean(gbsg$age)) / sd(gbsg$age)
gbsg$size = (gbsg$size - mean(gbsg$size)) / sd(gbsg$size)
gbsg$time = gbsg$rfstime

kaplan_meiers[[2]] <- get_kaplan_meier(gbsg)

#
# Save

write_json(kaplan_meiers, file.path("../benchmark_data/benchmark_kaplan_meier.json"))
