library(caret)
library(car)
library(MASS)
library(gvlma)
library(party)
library(RSNNS)
library(lmtest)
library(sandwich)
load("/project/msca/capstone3/patient_matrix2revise.RData")
load("/project/msca/kjtong/capstone.october.RData")
#load("/project/msca/capstone3/Octoberfinalmodels.RData")
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
zerovariance<-nearZeroVar(training_set, saveMetrics=TRUE)
#zv<-nearZeroVar(training_set, uniqueCut=, freqCut=9)
nrow(zerovariance[zerovariance[,"zeroVar"]=="TRUE",])
#zerovariance[zerovariance$nzv,][1:50,]
#zv<-subset(zerovariance, zeroVar=='TRUE')
#zv<-zerovariance[zerovariance[,"zeroVar"]=="TRUE",]
training_set<-training_set[,zerovariance$zeroVar==FALSE]
#zv<-apply(training_set,2,function(x) length(unique(x))==1) #code for zero variance predictors
#sum(zv)
#training_set<-training_set[,!zv] #remove zero variance predictors



ncol(training_set)
dfCorr<-cor(training_set) #build correlation matrix
highCorr<-findCorrelation(dfCorr, 0.7) #find all pairwise correlations greater than 0.7
traindf<-training_set[,-highCorr] #remove highly correlated predictors from training set
testdf<-testing_set 
ncol(traindf)
ncol(testdf)


#Build benchmark linear regression from reduced data and look at residuals to determine if further transformations are needed
fulllm<-lm(y2_charges~., data=traindf)
summary(fulllm)
fulllm.MSE <- mean((fulllm$fitted.values - traindf$y2_charges)^2)
fulllm.MSE


#heterosketaskity-robust standard errors

fulllm$robse<-vcovHC(fulllm, type="HC1")
coeftest<-coeftest(fulllm, fulllm$robse)
charges_hat<-fitted(fulllm) #predicted values
charges_hat_fram<-as.data.frame(charges_hat)
charges_resid<-residuals(fulllm)
charges_resid_frame<-as.data.frame(charges_resid)
residualplots<-residualPlot(fulllm)
avplot<-avPlots(fulllm, id.n=5, id.cex=0.7)
qqplot<-qqPlot(fulllm, id.n=5)
influenceindexplot<-influenceIndexPlot(fulllm, id.n=5)
influenceplot<-influencePlot(fulllm, id.n=5)
#Look at Residuals and outliers
outlier<-outlierTest(fulllm, cutoff=INF, n.max=INF)
#qqPlot(fulllm)
#leveragePlot(fulllm)
#av.Plots(fulllm)
sresid<-studres(fulllm)
hist(sresid, freq=FALSE)
curve(dnorm, add=TRUE)
#non-constant error variance test
ncvtest<-ncvTest(fulllm)
#Non-constant Variance Score Test 
#Variance formula: ~ fitted.values 
#Chisquare = 141101.3    Df = 1     p = 0 #data is significantly heteroscedastic
spreadLevelPlot(fulllm)
plot(density(resid(fulllm)))
y2_charges.res<-resid(fulllm)
plot(y2_charges.res)
abline(0,0)
y2_charges.stdres<-rstandard(fulllm)
qqnorm(y2_charges.stdres)
qqline(y2_charges.stdres)
hist(resid(fulllm))

#Evaluate nonlinearity
#crPlots(fulllm)
#ceresPlots(fulllm)
#Test for Autocorrelated Errors
durbinwatson<-durbinWatsonTest(fulllm)
#Global test for model assumptions
gvmodel<-gvlma(fulllm)
summary(gvmodel)

#Predict on test set
testlm<-lm(y2_charges~., data=testdf)
pred<-predict(fulllm, newdata=testdf)
lmRMSE<-sqrt(mean((pred- testdf$y2_charges)^2))
lmRMSE 
lmMAE<-mean(abs(pred- testdf$y2_charges))
lmMAE



#Look at predictors correlation to Y

#Perform necessary transformations

#build linear model again
#look at residuals again after transformations
#Perform PCA
set.seed(10)
patient_matrixpca<-patient_matrix

patient_matrixpca$patient_id<-NULL
patient_matrixpca$mode_discharge_dispo<-NULL
patient_matrixpca$sex<-NULL
patient_matrixpca$race<-NULL
patient_matrixpca$ethnicity<-NULL
patient_matrixpca$marital_status<-NULL
trainpca<-createDataPartition(patient_matrixpca$y2_charges, p=0.7, list=FALSE)
#train<- sample(1:nrow(df),floor(0.7*nrow(df)))
training_set_pca <- patient_matrixpca[trainpca,]
testing_set_pca <- patient_matrixpca[-trainpca,]



zvpca<-nearZeroVar(training_set_pca, saveMetrics=TRUE)

nrow(zvpca[zvpca[,"zeroVar"]=="TRUE",])

training_set_pca<-training_set_pca[,(zvpca$zeroVar==FALSE)]

training_set_pca$y2_charges<-NULL


classes<-list()
for (i in 1:ncol(training_set_pca)){
  classes[i]<-class(training_set_pca[,i])
}
test_int<-training_set_pca[,c(which(classes=="integer"))]
train_num1<-training_set_pca[,c(which(classes=="numeric"))]

train_num2<-apply(train_int,2,as.numeric)
train_num<-cbind(train_num1, as.data.frame(train_num2))

PCA<-preProcess(train_num,method=c("pca"))


#bind back y variable


#re-run linear models
fulllmpca<-lm(y2_charges~., data=xtrain)  #what do you use as the response variable?
summary(fulllmpca)
fulllm.MSE.pca <- mean((fulllmpca$fitted.values - xtrain$y2_charges)^2) #response variable?
fulllm.MSE.pca
#Machine learning models
#regression tree
dftree<-patient_matrix
dftree$patient_id<-NULL
set.seed(10)

traintree <- createDataPartition(dftree$y2_charges, p=0.7, list=FALSE)
training_set_tree<- dftree[traintree,]
testing_set_tree <- dftree[-traintree,]
x1_train <- model.matrix(y2_charges~.,training_set_tree)
x1_test <- model.matrix(y2_charges~.,testing_set_tree)
y1_train <- training_set_tree$y2_charges
y1_test <- testing_set_tree$y2_charges

formula<-y2_charges~.

fit.ctree.party<-ctree(formula, data=training_set_tree, controls=ctree_control(mincriterion=0.95,savesplitstats=FALSE))
fit.ctree.party

#bootControl<-trainControl(method="cv", number=6, summaryFunction=defaultSummary)
#set.seed(10)
#getModelInfo("ctree2", FALSE)[[1]]$grid
#modelLookup("ctree2")
#Grid<-expand.grid(maxdepth=seq())
#treefit<-train(x1_train, y1_train,method="rpart2", maxdepth=3)
#treefit
#treefit$finalModel

getModelInfo("rpart", FALSE)[[1]]$grid
modelLookup("rpart")
treefit<-train(x1_train, y1_train,method="rpart2", maxdepth=3)
min(treefit$resample[1])
#nncontrol<-trainControl(method="LOOCV", seeds=seeds)
getModelInfo("mlp", FALSE)[[1]]$grid
modelLookup("mlp")
nnet<-train(x1_train, y1_train,method="mlp", size=4)


#boosted tree

getModelInfo("gbm", FALSE)[[1]]$grid
modelLookup("gbm")
control<-trainControl(method="repeatedCV", number=10, repeats=1, verboseIter=FALSE, returnResamp="all", classProbs=TRUE,)
gbmGrid<-expand.grid(interaction.depth=seq(1,4), n.trees=(1:4)*50, shrinkage=.1, n.minobsinnode=10)
set.seed(10)
gbmFit<-train(x1_train, y1_train, method="gbm", trControl=control, tuneGrid=gbmGrid, metric="RMSE", verbose=FALSE)
gbmFit
#Conduct visualizations

#Show model comparisons

#predict of new samples
predValuesparty<-predict(fit.ctree.party, newdata=testing_set_tree)
RMSE.test.party<-RMSE(obs=y1_test, pred=predValuesparty)
RMSE.test.party
partymae<-mean(abs(predValuesparty-y1_test))
partymae


predValuesGBM<-predict(gbmFit, testX=x1_test, testY=y1_test)
RMSE.test.gbm<-RMSE(obs=y1_test, pred=predValuesGBM)
RMSE.test.gbm
gbmmae<-mean(abs(predValuesGBM-y1_test))
gbmmae


predValuestree<-predict(treefit, testX=x1_test, testY=y1_test)
RMSE.test.tree<-RMSE(obs=y1_test, pred=predValuestree)
RMSE.test.tree
treemae<-mean(abs(predValuestree-y1_test))
treemae






#trials
testValues<-subset(predValues, dataType=="Test")
table(testValues$model)
rpart2Pred<-subset(testValues, model=="rpart2")
rpart2Pred

testpredict<-predict(models, newdata=testing_set_tree)
predict(treefit$finalModel, newdata=testing_set_tree)

predict(treefit, newdata=testing_set_tree)

models<- list(ctree=treefit, gbm=gbmFit)
testPred<-predict(models, newdata=testing_set_tree)
lapply(testPred, function(x) x[1:5])

predValues<-extractPrediction(models, testX=testing_set_tree, testY= testClass)
testValues<-subset(predValues, dataType=="Test")
head(testValues)
table(testValues$model)







#str(dftree)




regression.tree<-tree(y2_charges~.,dftree,subset=traintree)
summary(regression.tree)

plot(regression.tree)
text(regression.tree,pretty=0)
cv.regressiontree=cv.tree(regression.tree)
plot(cv.regressiontree$size,cv.regressiontree$dev,type="b")

prune.regressiontree=prune.tree(regression.tree,best=7)
plot(prune.regressiontree)
text(prune.regressiontree,pretty=0)

yhat=predict(regression.tree,newdata=testing_set_tree)
regression.test=dftree[-traintree, "y2_charges"]
plot(yhat,regression.test)
abline(0,1)
mean((yhat-regression.test)^2)


