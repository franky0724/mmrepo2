library(rpart)

rScoreFunction <- function(JOB, REASON, CLAGE, CLNO, DEBTINC, DELINQ, DEROG, NINQ, YOJ)
{
   #output: EM_EVENTPROBABILITY, EM_CLASSIFICATION

   if (!exists("myClassTree"))
   {
      assign("myClassTree", readRDS(file = paste(rdsPath, 'hmeq_classtree_r.rds', sep = '')), envir = .GlobalEnv)
   }

   # Threshold for the misclassification error
   threshPredProb <- 0.199916001679966

   # Recode JOB into a numeric factor
   JOBtype <- 0
   JOBtype[JOB == 'Mgr'] <- 1
   JOBtype[JOB == 'Office'] <- 2
   JOBtype[JOB == 'Other'] <- 3
   JOBtype[JOB == 'ProfExe'] <- 4
   JOBtype[JOB == 'Sales'] <- 5
   JOBtype[JOB == 'Self'] <- 6

   # Recode REASON into a numeric factor
   REASONtype <- 0
   REASONtype[REASON == 'DebtCon'] <- 1
   REASONtype[REASON == 'HomeImp'] <- 2

   # Impute missing covariates by their means
   CLAGE[is.na(CLAGE) || is.null(CLAGE)] <- 178.6067620442
   CLNO[is.na(CLNO) || is.null(CLNO)] <- 21.21746724891
   DEBTINC[is.na(DEBTINC) || is.null(DEBTINC)] <-  33.47332952233

   # Impute missing covariates by their modes
   DELINQ[is.na(DELINQ) || is.null(DELINQ)] <- 0.0
   DEROG[is.na(DEROG) || is.null(DEROG)] <- 0.0
   NINQ[is.na(NINQ) || is.null(NINQ)] <- 0.0
   YOJ[is.na(YOJ)] <- 7.0

   input_array <- data.frame('JOBtype' = as.factor(JOBtype), 'REASONtype' = as.factor(REASONtype),
                             'CLAGE' = CLAGE, 'CLNO' = CLNO, 'DEBTINC' = DEBTINC,
                             'DELINQ' = DELINQ, 'DEROG' = DEROG, 'NINQ' = NINQ, 'YOJ' = YOJ)

   predProb <- predict(myClassTree, newdata = input_array, type='prob', na.action = na.omit)

   # Retrieve the event probability
   EM_EVENTPROBABILITY <- predProb[2]

   # Determine the predicted target category
   EM_CLASSIFICATION <- ifelse(EM_EVENTPROBABILITY >= threshPredProb, '1', '0')

   output_list <- list('EM_EVENTPROBABILITY' = EM_EVENTPROBABILITY, 'EM_CLASSIFICATION' = EM_CLASSIFICATION)
   return(output_list)
}
