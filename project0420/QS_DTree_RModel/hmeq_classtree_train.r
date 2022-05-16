# title: 'hmeq_classtree_train.r'
# author: 'SAS Institute'
# date: 'July 2, 2020'

# Control to print all 13 digits
options(digits = 13)

# Load the necessary libraries
library(dplyr)
library(rpart)

# Specify location of the analysis folder
analysisFolder = 'C:\\MyJob\\Projects\\ModelManager\\Test\\HMEQ\\'
analysisPrefix = 'hmeq_classtree'

# Specify location of the JSON folder
jsonFolder = 'C:\\MyJob\\Projects\\ModelManager\\Test\\HMEQ\\ClassTree\\R\\'

# Specify location of the custom source folder
sourceFolder = 'C:\\Users\\minlam\\Documents\\Research\\R\\'

# Bring in custom codes for exporting model
source(paste(sourceFolder, 'export_binary_model.r', sep = ''))

# Read the CSV data, and recast JOB and REASON to just Characters because they came as factors
inputData <- read.table(paste(analysisFolder, 'hmeq_train.csv', sep = ''), header = TRUE, sep = ',')
inputData[c('JOB', 'REASON')] = lapply(inputData[c('JOB', 'REASON')], as.character)

# Recode JOB into a numeric factor
inputData$JOBtype <- 0
inputData$JOBtype[inputData$JOB == 'Mgr'] <- 1
inputData$JOBtype[inputData$JOB == 'Office'] <- 2
inputData$JOBtype[inputData$JOB == 'Other'] <- 3
inputData$JOBtype[inputData$JOB == 'ProfExe'] <- 4
inputData$JOBtype[inputData$JOB == 'Sales'] <- 5
inputData$JOBtype[inputData$JOB == 'Self'] <- 6

# Recode REASON into a numeric factor
inputData$REASONtype <- 0
inputData$REASONtype[inputData$REASON == 'DebtCon'] <- 1
inputData$REASONtype[inputData$REASON == 'HomeImp'] <- 2

# Specify the target variables, the nominal predictors, and the interval predictors
targetVar <- 'BAD'
nominalVars <- c('JOBtype', 'REASONtype')
intervalVars <- c('CLAGE', 'CLNO', 'DEBTINC', 'DELINQ', 'DEROG', 'NINQ', 'YOJ')
myvars <- c(targetVar, nominalVars, intervalVars)

# Remove all observations where the target variable is missing
trainData <- inputData[!is.na(inputData$BAD), myvars]

# Get threshold for the misclassification error
threshPredProb <- mean(trainData$BAD, na.rm = TRUE)
print(paste('Observed Probability that BAD=1 is', threshPredProb))

for (col in c(targetVar,nominalVars))
{
    trainData[[col]] <- as.factor(trainData[[col]])
    print(col)
    print(levels(trainData[[col]]))
}

# Although we will use only complete cases for training the model, we need estimates of the variables' location for imputation for scoring

# Check for number of missing values in each variable in hmeq
sapply(trainData, function(x) sum(is.na(x)))

# Get the means of CLAGE and CLNO
print('Means on Complete Cases:')
print(sapply(trainData[c('CLAGE','CLNO')], function(x) mean(x, na.rm = TRUE)))

# Get the medians of DEBTINC, DELINQ, DEROG, NINQ, and YOJ
print('Medians on Complete Cases:')
print(sapply(trainData[c('DEBTINC','DELINQ','DEROG','NINQ','YOJ')], function(x) median(x, na.rm = TRUE)))

#Get the modes of JOB and REASON
Mode <- function(x) {
  t <- table(x)
  names(t)[ which(t == max(t)) ]
}

print('Modes on Complete Cases:')
print(sapply(trainData[c('JOBtype','REASONtype')], function(x) Mode(x)))

# Omit all the rows which have missing values
trainData = na.omit(trainData)
print(head(trainData))

# Get ready for training the classification tree model.

# A classification tree, do not carry competitors along, no surrogate, no cross-validation
# split if more than 50 observations, maximum depth is 10
myClassTree <- rpart(BAD ~ JOBtype + REASONtype + CLAGE + CLNO + DEBTINC + DELINQ + DEROG + NINQ + YOJ,
                     method = 'class', data = trainData, na.action = na.omit,
                     control = rpart.control(maxcompete = 0, maxsurrogate = 0, xval = 0,
                                             minsplit = 25, maxdepth = 10))

# Display the tree diagram
library(rpart.plot)
rpart.plot(myClassTree)
rpart.rules(myClassTree)

# See the model fit summary
print(summary(myClassTree))

# Calculate the prediction and save then to an external CSV file

# See the Misclassification Rate of the training data
fitted.prob <- predict(myClassTree, newdata = trainData, type='prob')
fitted.results <- ifelse(fitted.prob[,2] >= threshPredProb, 1, 0)

# Save the predictions to CSV
write.csv(cbind(trainData, fitted.prob, fitted.results), paste(analysisFolder, analysisPrefix, '_r_pred.csv', sep = ''))

# Lengths of original nominal predictors
nominalVarsLength <- as.data.frame(do.call(rbind, lapply(inputData[,c('JOB', 'REASON')], function(x) max(nchar(x)))))
print(nominalVarsLength)

# Types of the columns
typeOfColumn <- as.data.frame(do.call(rbind, lapply(inputData, typeof)))
print(typeOfColumn)

# Invoke this function to generate the zip package
export_binary_model (
    targetVar = targetVar,
    intervalVars = intervalVars,
    nominalVars = c('JOB', 'REASON'),
    nominalVarsLength = nominalVarsLength,
    typeOfColumn = typeOfColumn,
    targetValue = trainData$BAD,
    eventValue = 1,
    predEventProb = fitted.prob[,2],
    eventProbThreshold = threshPredProb,
    algorithmCode = 3,
    modelObject = myClassTree,
    analysisFolder = analysisFolder,
    analysisPrefix = analysisPrefix,
    jsonFolder = jsonFolder,
    analysisName = 'Home Mortgage Equity Qualification (R)',
    analysisDescription = 'rpart.control(maxcompete=0, maxsurrogate=0, xval=0, minsplit=25, maxdepth=10)',
    qDebug = 'Y')

# THE END
