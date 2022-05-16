import numpy
import pandas
import settings

from rpy2 import robjects
from rpy2.robjects.packages import importr

base = importr('base')
rpart = importr('rpart')

# Get the predicted event probability
rstring = """
    function(myClassTree, JOB, REASON, CLAGE, CLNO, DEBTINC, DELINQ, DEROG, NINQ, YOJ)
    {
        # Threshold for the misclassification error
        threshPredProb <- 0.199916001679966

        # Recode JOB into a numeric factor
        JOBtype <- 0
        JOBtype[JOB == "Mgr"] <- 1
        JOBtype[JOB == "Office"] <- 2
        JOBtype[JOB == "Other"] <- 3
        JOBtype[JOB == "ProfExe"] <- 4
        JOBtype[JOB == "Sales"] <- 5
        JOBtype[JOB == "Self"] <- 6

        # Recode REASON into a numeric factor
        REASONtype <- 0
        REASONtype[REASON == "DebtCon"] <- 1
        REASONtype[REASON == "HomeImp"] <- 2

        # Impute missing covariates by their means
        CLAGE[is.na(CLAGE) || is.null(CLAGE)] <- 178.6067620442
        CLNO[is.na(CLNO) || is.null(CLNO)] <- 21.21746724891
        DEBTINC[is.na(DEBTINC) || is.null(DEBTINC)] <-  33.47332952233

        # Impute missing covariates by their modes
        DELINQ[is.na(DELINQ) || is.null(DELINQ)] <- 0.0
        DEROG[is.na(DEROG) || is.null(DEROG)] <- 0.0
        NINQ[is.na(NINQ) || is.null(NINQ)] <- 0.0
        YOJ[is.na(YOJ)] <- 7.0

        input_array <- data.frame("JOBtype" = as.factor(JOBtype), "REASONtype" = as.factor(REASONtype),
                                  "CLAGE" = CLAGE, "CLNO" = CLNO, "DEBTINC" = DEBTINC,
                                  "DELINQ" = DELINQ, "DEROG" = DEROG, "NINQ" = NINQ, "YOJ" = YOJ)

        predProb <- predict(myClassTree, newdata = input_array, type='prob', na.action = na.omit)

        # Retrieve the event probability
        EM_EVENTPROBABILITY <- predProb[2]

        # Determine the predicted target category
        EM_CLASSIFICATION <- ifelse(EM_EVENTPROBABILITY >= threshPredProb, "1", "0")

        output_list <- list("EM_EVENTPROBABILITY" = EM_EVENTPROBABILITY, "EM_CLASSIFICATION" = EM_CLASSIFICATION)
        return(output_list)
    }
"""

# Instantiate the r function
rScoreFunction = robjects.r(rstring)

def scoreHMEQClassTreeModel (JOB, REASON, CLAGE, CLNO, DEBTINC, DELINQ, DEROG, NINQ, YOJ):
    "Output: EM_EVENTPROBABILITY, EM_CLASSIFICATION"

    # Retrieve the R model object as if it is done by the R software
    global myClassTree
    try:
        myClassTree
    except NameError:
        myClassTree = base.readRDS(file = base.paste(settings.pickle_path, "hmeq_classtree_r.rds", sep = ""))

    # Handle missing values
    try:
        _JOB = JOB.strip()
    except AttributeError:
        _JOB = ""

    try:
        _REASON = REASON.strip()
    except AttributeError:
        _REASON = ""

    try:
        _CLAGE = CLAGE + 0.0
    except TypeError:
        _CLAGE = numpy.nan

    try:
        _CLNO = CLNO + 0.0
    except TypeError:
        _CLNO = numpy.nan

    try:
        _DEBTINC = DEBTINC + 0.0
    except TypeError:
        _DEBTINC = numpy.nan

    try:
        _DELINQ = DELINQ + 0.0
    except TypeError:
        _DELINQ = numpy.nan

    try:
        _DEROG = DEROG + 0.0
    except TypeError:
        _DEROG = numpy.nan

    try:
        _NINQ = NINQ + 0.0
    except TypeError:
        _NINQ = numpy.nan

    try:
        _YOJ = YOJ + 0.0
    except TypeError:
        _YOJ = numpy.nan

    rOutput = rScoreFunction (myClassTree, _JOB, _REASON, _CLAGE, _CLNO, _DEBTINC, _DELINQ, _DEROG, _NINQ, _YOJ)
    
    EM_EVENTPROBABILITY = rOutput[0][0]
    EM_CLASSIFICATION = rOutput[1][0]

    return(EM_EVENTPROBABILITY, EM_CLASSIFICATION)
