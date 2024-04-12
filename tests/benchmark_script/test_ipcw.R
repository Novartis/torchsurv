library(pec)

get_ipcw <- function(data){
  
  sorted_indices <- order(data$time)
  time = data$time[sorted_indices]
  status = data$status[sorted_indices]
  times <- quantile(time, probs=seq(0.2,0.8,0.1))
  
  ipcw <- pec::ipcw(Surv(failure_time,status)~1,
                       data=data.frame(failure_time=time,status=as.numeric(status!=0)),
                       method="marginal",
                       times=times,
                       subjectTimes=T,
                       subjectTimesLag=0)
  
  ipcw_times <- 1/ipcw$IPCW.times
  ipcw_subjectimes <- 1/ipcw$IPCW.subjectTimes
  ipcw_subjectimes[ipcw_subjectimes == Inf] = 0
  
  return(list(time=time, status=status, times=times, ipcw_times=ipcw_times, ipcw_subjectimes=ipcw_subjectimes))
}

# create list to save aucs
ipcws <- list()
i = 1

#
# lung dataset 

lung <- survival::lung
lung$sex = lung$sex  == 1
lung$status = lung$status  == 2
lung$age = (lung$age - mean(lung$age)) / sd(lung$age)

ipcws[[1]] <- get_ipcw(lung)

#
# gbsg dataset 

gbsg <- survival::gbsg
gbsg$age = (gbsg$age - mean(gbsg$age)) / sd(gbsg$age)
gbsg$size = (gbsg$size - mean(gbsg$size)) / sd(gbsg$size)
gbsg$time = gbsg$rfstime

ipcws[[2]] <- get_ipcw(gbsg)


#
# Save

write_json(ipcws, file.path("../benchmark_data/benchmark_ipcw.json"))


