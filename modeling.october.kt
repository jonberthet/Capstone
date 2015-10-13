library(caret)
library(car)
library(MASS)
library(gvlma)
load("/project/msca/capstone3/patient_matrix2revise.RData")

#Test and Train for Multicollinearity on numerical variables
set.seed(10)
df<-patient_matrix
df$patient_id<-NULL
df$mode_discharge_dispo<-NULL
df$sex<-NULL
df$race<-NULL
df$ethnicity<-NULL
df$marital_status<-NULL
train<-createDataPartition(df$y2_charges, p=0.7, list=FALSE)
#train<- sample(1:nrow(df),floor(0.7*nrow(df)))
training_set <- df[train,]
testing_set <- df[-train,]

#Multicollinearity of numeric variables and removal of highly correlated variables
zv<-apply(training_set,2,function(x) length(unique(x))==1) #code for zero variance predictors
sum(zv)
training_set<-training_set[,!zv] #remove zero variance predictors

ncol(training_set)
dfCorr<-cor(training_set) #build correlation matrix
highCorr<-findCorrelation(dfCorr, 0.7) #find all pairwise correlations greater than 0.7
traindf<-training_set[,-highCorr] #remove highly correlated predictors from training set
testdf<-testing_set[,-highCorr] #remove highly correlated predictors from testing set
ncol(traindf)
ncol(testdf)

#Build benchmark linear regression from reduced data and look at residuals to determine if further transformations are needed
fulllm<-lm(y2_charges~., data=traindf)
summary(fulllm)
fulllm.MSE <- mean((fulllm$fitted.values - traindf$y2_charges)^2)
fulllm.MSE

#Look at Residuals and outliers
outlierTest(fulllm)
#qqPlot(fulllm)
#leveragePlot(fulllm)
#av.Plots(fulllm)
sresid<-studres(fulllm)
hist(sresid, freq=FALSE)
curve(dnorm, add=TRUE)

ncvTest(fulllm)
spreadLevelPlot(fit)
plot(density(resid(fulllm)))
y2_charges.res<-resid(fulllm)
plot(y2_charges.res)
abline(0,0)
y2_charges.stdres<-rstandard(fulllm)
qqnorm(y2_charges.stdres)
qqline(y2_charges.stdres)
hist(resid(fulllm))
#Double check no multicollinearity
vif(fulllm)
sqrt(vif(fulllm))>2
#Evaluate nonlinearity
crPlots(fulllm)
ceresPlots(fulllm)
#Test for Autocorrelated Errors
durbinWatsonTest(fulllm)
#Global test for model assumptions
gvmodel<-gvlma(fulllm)
summary(gvmodel)
#Predict on test set
pred<-predict(fulllm, newdata=testdf)
summary(pred)

#Look at predictors correlation to Y

#Perform necessary transformations

#Perform PCA
set.seed(10)
xTrans<-preProcess(traindf, method=c("center","scale", "pca"))

xtrain<-predict(xTrans, traindf)
xtest<-predict(xTrans, testdf)
