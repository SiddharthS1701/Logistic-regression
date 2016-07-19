### Steps to perform logistic regression

# Preparation for analysis

# set the working directory
# -------------------------
setwd("C:/Users/siddharth.s/Desktop/01.Logistic regression/directory")

# check the working directory
# ---------------------------
getwd()

# install the required packages
# -----------------------------
install.packages("aod")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("glmulti")
install.packages("plyr")
install.packages("lattice")
install.packages("ellipse")
install.packages("corrplot")
install.packages("foreach", dependencies = T)
install.packages("caret")
install.packages("mlbench")
install.packages("Hmisc", dependencies = T)

# load the required packages
# --------------------------
library(aod)
library(ggplot2)
library(dplyr)
library(glmulti)
library(plyr)
library(lattice)
library(ellipse)
library(corrplot)
#library(iterators)
#library(foreach)
#library(caret)
library(mlbench)
library(Hmisc)

##############################################################
####1 import the analytical dataset and conducted basic eda
##############################################################

# read the dataset into R
# ------------------------
mydata <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
write.csv(mydata, "mydata.csv")
str(mydata)


# Data dictionary
  # admit <- flag indicating admission into graduate school
  # rank <- prestige of the undergraduate institution#
  # gre <- score from a standardized test
  # gpa <- undergraduate course scores


# lets add some additional variables into our data
# ------------------------------------------------
mydata$admit_n <- as.factor(mydata$admit)
mydata$rank <- as.factor(mydata$rank)

# structure of the base dataset
# colnames
# ------------------------------
col.names <- as.data.frame(colnames(mydata))

# structure of the variables along with the dataset name
# ------------------------------------------------------
str(mydata)

# summary statistics
# ------------------
summary(mydata)
plot(mydata$admit) # distribution of the dependent variable (y)

# histogram for checking distribution
# -----------------------------------
histogram(~mydata$admit, mydata) # y variable
histogram(~mydata$rank, mydata)
histogram(~mydata$gre, mydata)
histogram(~mydata$gpa, mydata)

# box plots for continuous variable - check for outliers
# ------------------------------------------------------
bwplot(~mydata$gre, mydata)
bwplot(~mydata$gpa, mydata)

# cross tabs for categorical variables - equivalent to proc freq in sas
# ---------------------------------------------------------------------
xtabs(~admit + rank, data = mydata)
admitVsRank <- as.data.frame(xtabs(~admit + rank, data = mydata))


# # table df option
# mydata.tbldf <- tbl_df(mydata)
# mydata.tbldf

# correlation matrix
# ------------------
mydata$gre_num <- as.numeric(mydata$gre)

ctab <- cor(select(mydata,-gre,-rank,-admit, -admit_n))
round(ctab,2)

# Make the graph, with reduced margins
# ------------------------------------
plotcorr(ctab, mar = c(0.1, 0.1, 0.1, 0.1))

# graphical correlation matrix with scatter plots
# -----------------------------------------------
corrplot(ctab, method = "circle")  # explore the corrplot package for different representations
corrplot(ctab, method = "square")
corrplot(ctab, method = "ellipse")
corrplot(ctab, method = "number")
corrplot(ctab, method = "color")
corrplot(ctab, type = "lower")

# frequency distribution for a variable
#--------------------------------------
Freq <- table(mydata$admit)

z <- as.data.frame(Freq)
mutate(z, relFreq = prop.table(Freq), Cumulative_Freq = cumsum(Freq), 
       Cumulative_Relative_Freq = cumsum(relFreq))


##############################################################
####2 Divide the model into training and test data
##############################################################

# method 1
# 
# set.seed(007) # helps to generate a unique sample which is reproducible with same seed value
# inTrain <- createDataPartition(y = mydata$admit,
#                                            ## the outcome data are needed
#                                            p = .75,
#                                            ## The percentage of data in the
#                                            ## training set
#                                            list = FALSE)
#                                            ## The format of the results
# 
# ## the output is a set of integers for the rows of mydata
# ## that belong to training data
# str(inTrain)
# 
# nrows(mydata) # check the number of rows in the original dataset
# 
# mydata.train <- mydata[inTrain,]
# nrow(mydata.train) # check the number of rows in the training dataset
# 
# mydata.test <- mydata[-inTrain,]
# nrow(mydata.test) # check the number of rows in the test dataset

# method 2

## 75% of the sample size
smp_size <- floor(0.75 * nrow(mydata))

## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(mydata)), size = smp_size)

# training dataset    
mydata.train <- mydata[train_ind, ]
# test dataset
mydata.test <- mydata[-train_ind, ]


##############################################################
####3 check the model for outliers / missing values
##############################################################

# http://www.r-bloggers.com/finding-outliers-in-numerical-data/

# box plots for continuous variable - check for outliers
#-------------------------------------------------------
bwplot(~mydata$gre, mydata)
bwplot(~mydata$gpa, mydata)

###########################################################################
####4 Run model iterations using glm() function and find the model equation
###########################################################################

# Start by building the global or the full model and the model without any variables
# ----------------------------------------------------------------------------------
mylogit <- glm(admit_n ~ gre + rank + gpa, data = mydata.train, family = binomial)
summary(mylogit)

nothing <- glm(admit_n ~ 1, data=mydata.train, family =  binomial)
summary(nothing)

# generate odds ratio
exp(coef(mylogit))

# generate odds ratio along with the confidence intervals
exp(cbind(OR = coef(mylogit), confint(mylogit)))

# variable selection algorithms
# -----------------------------

# backward  - use the full model as the parameter for the step function - backward selection is the default
backward.mylogit <- step(mylogit)
# backward.mylogit <- step(mylogit, trace = 0) # suppress the output

# extract the final equation
formula(backward.mylogit)

rm(raw_data_1)
# forward
forward.mylogit <- step(nothing, scope=list(lower=formula(nothing), upper=formula(mylogit)), direction="forward")
formula(forward.mylogit)

# bothways
bothways.mylogit <- step(nothing, list(lower=formula(nothing),upper=formula(mylogit)), direction="both",trace=0)
formula(bothways.mylogit)

# algorithms with glmulti package
best.mylogit <- glmulti(mylogit, level = 1, crit="aic", family = binomial) # use AICc when working with small data

summary(best.mylogit)
bestmodel <- summary(best.mylogit)$bestmodel

bm <- as.data.frame(weightable(best.mylogit))
bm_1 <- as.character(bm[1,1])

#final model
train_model <- glm(as.formula(bm_1), data = mydata.train, family = binomial)
summary(train_model)

# Getting the std estimates
# --------------------------
install.packages("QuantPsyc")
library(QuantPsyc)
lm.beta(train_model)

##############################################################
####5 Predict / score the test dataset and compute model diagnostics
##############################################################

#a actual vs predicted

# in sample
mydata.train$Probscore <- predict(train_model, newdata = mydata.train, type = "response")
mydata.train  

# create  bins for predicted score
mydata.train$pred_group <- as.numeric(cut2(mydata.train$Probscore, g=10)) # cut2 allows split based on value of a column
unique(mydata.train$pred_group)

mydata.train$admit_n <- as.numeric(mydata.train$admit)
mydata.train$pred_group_n <- as.factor(mydata.train$pred_group)

detach(package:plyr)

# function to generate actual probability and mean (predicted probability) for each decile

#act_pred$admit_num <- as.numeric(act_pred$admit)

act_pred <- 
  mydata.train %>%
  group_by(pred_group_n) %>%
  summarise(
    Predicted = mean(Probscore, na.rm = TRUE),
    Actual = sum(admit),
    Count = n()
  )

# create the new actual variable
# ------------------------------
act_pred$Actual_f <- act_pred$Actual/act_pred$Count 

# create the actual vs. predicted graph
# -------------------------------------
ggplot(act_pred, aes(as.numeric(pred_group_n))) + 
  geom_line(aes(y = Actual_f, colour = "Actual_f")) + 
  geom_line(aes(y = Predicted, colour = "Predicted")) + ggtitle("Actual vs. Predicted")

#b roc curve & senstivity vs specificity
# --------------------------------------
install.packages("pROC")
library(pROC)
g <- roc(admit_n ~ Probscore, data = mydata.train)
plot(g) 

# Extract the best probability value for cut off
coords(g, "best")

# create prediction flags based on the probability cut off
mydata.train$prediction=ifelse(mydata.train$Probscore>=0.3659147,1,0)

#d confusion matrix and other parameters

cf_matrix <- xtabs(~admit_n + prediction, data = mydata.train)

confusionmatrix <- function(d){
  total_count <- d[1,1] + d[1,2] + d[2,1] + d[2,2]
  accuracy <- (d[1,1] + d[2,2])/total_count # percentage of 1s and 0s correctly predicted
  precision <- d[2,2]/(d[1,2] + d[2,2]) # percentage of correct 1s correctly predicted
  hitrate <- d[2,2]/(d[2,1] + d[2,2]) # percentage of actual 1s correctly captured
  cfmatrix <- c(accuracy, precision, hitrate)
  return(cfmatrix)
}

confusionmatrix(cf_matrix)

# Generate the final model results along with std. betas to compare predictors
# ----------------------------------------------------------------------------

act_pred$act_sum <-(act_pred$Actual/sum(act_pred$Actual))*100
act_pred <- arrange(act_pred, desc(pred_group_n))
act_pred <- act_pred %>% mutate(cumsum = cumsum(act_sum))

random <- c(seq(10,100,10)) # generate a new column that stands for random targetting
deciles <- c(seq(1,10,1))
act_pred <- data.frame(act_pred,random, deciles) #adding back to the data frame

# plot the gains chart

ggplot(act_pred, aes(deciles)) +
  geom_line(aes(y = random, colour = "random")) +
  geom_line(aes(y = cumsum, colour = "cumsum")) + ggtitle("Gains Chart")

# plot the logistic regression S curve

#ggplot(mydata.train, aes(x=gre, y=admit)) + geom_point() + 
#  stat_smooth(method="glm", family="binomial",se=FALSE)




