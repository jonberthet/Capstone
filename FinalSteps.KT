library(plyr)
library(caret)
library(e1071)
library(kernlab)
load("/project/msca/capstone3/patient_matrix2revise.RData")
load("/project/msca/capstone3/Octoberfinalmodels.RData")
#Exploratory analysis of charges
nrow(patient_matrix)#44011 observations
hist(patient_matrix$y2_charges)
zero<-subset(patient_matrix, y2_charges==0) #10008 observations; 22.7% of patients
summary(zero$race)/sum(summary(zero$race))#Percentages of different races within subset
summary(zero$marital_status)/sum(summary(zero$marital_status))
summary(zero$sex)/sum(summary(zero$sex))
summary(zero$ethnicity)/sum(summary(zero$ethnicity))
low<-subset(patient_matrix, y2_charges<1000 & y2_charges>0)#6347 observations; 14.4% of patients
hist(low$y2_charges)
summary(low$race)/sum(summary(low$race))#Percentages of different races within subset
summary(low$marital_status)/sum(summary(low$marital_status))
summary(low$sex)/sum(summary(low$sex))
summary(low$ethnicity)/sum(summary(low$ethnicity))
low2<-subset(patient_matrix, y2_charges<10000 & y2_charges>1000)#16120; 36.6% of patients
hist(low2$y2_charges)
summary(low2$race)/sum(summary(low2$race))#Percentages of different races within subset
summary(low2$marital_status)/sum(summary(low2$marital_status))
summary(low2$sex)/sum(summary(low2$sex))
summary(low2$ethnicity)/sum(summary(low2$ethnicity))
med<-subset(patient_matrix, y2_charges<500000 & y2_charges>10000)#11470; 26.06% of patients
hist(med$y2_charges)
summary(med$race)/sum(summary(med$race))#Percentages of different races within subset
summary(med$marital_status)/sum(summary(med$marital_status))
summary(med$sex)/sum(summary(med$sex))
summary(med$ethnicity)/sum(summary(med$ethnicity))
high<-subset(patient_matrix, y2_charges>500000)#65 obs; 1.47% of patients
hist(high$y2_charges)
high<-high[,2:15]
summary(high$race)/sum(summary(high$race))#Percentages of different races within subset
summary(high$marital_status)/sum(summary(high$marital_status))
summary(high$sex)/sum(summary(high$sex))
summary(high$ethnicity)/sum(summary(high$ethnicity))
racesummary<-ddply(high, .(race), summarise, meanlab=mean(labfreq), meanmed=mean(medfreq), meansumlos=mean(sumlos), meanencount=mean(encountfreq), meanhosp=mean(hospitalizations), meanER=mean(num_ER_visits), meany1charges=mean(y1_charges), meanage=mean(age_y2))
racesummary
maritalsummary<-ddply(high, .(marital_status), summarise, meanlab=mean(labfreq), meanmed=mean(medfreq), meansumlos=mean(sumlos), meanencount=mean(encountfreq), meanhosp=mean(hospitalizations), meanER=mean(num_ER_visits), meany1charges=mean(y1_charges), meanage=mean(age_y2))
maritalsummary
sexsummary<-ddply(high, .(sex), summarise, meanlab=mean(labfreq), meanmed=mean(medfreq), meansumlos=mean(sumlos), meanencount=mean(encountfreq), meanhosp=mean(hospitalizations), meanER=mean(num_ER_visits), meany1charges=mean(y1_charges), meanage=mean(age_y2))
sexsummary

#create train, test, validate sets

set.seed(10)
train <- sample(1:nrow(patient_matrix),floor(0.6*nrow(patient_matrix)))
training.set<-patient_matrix[train,]
totalvalidation.set<-patient_matrix[-train,]

invalidation<-sample(1:nrow(totalvalidation.set),floor(0.5*nrow(totalvalidation.set)))
testing.set<-totalvalidation.set[invalidation,]
validation.set<-totalvalidation.set[-invalidation,]
save(training.set, testing.set, validation.set,file="/project/msca/capstone3/datasplits.RData")


#Create clusters on training set
set.seed(10)
km.charges <- kmeans(data.matrix(training.set[,10]), 2, nstart=50)
plot(data.matrix(training.set[,10]), col = (km.charges$cluster+1), main = "Kmeans Clusters k=3", pch=20, cex=1)
km.charges$centers

training.set1<-cbind(training.set, chargesClustered = km.charges$cluster)

col_idx<-grep("chargesClustered",names(training.set1))
training.set1<-training.set1[,c(col_idx,(1:ncol(training.set1))[-col_idx])]

#create dummy variables for clusters
for(t in unique(training.set1$chargesClustered)){training.set1[paste("chargesClustered", t, sep="")]<-ifelse(training.set1$chargesClustered==t,1,0)}


#re-order variables in data frame
col_idx1<-grep("chargesClustered1",names(training.set1))
training.set1<-training.set1[,c(col_idx1,(1:ncol(training.set1))[-col_idx1])]

col_idx2<-grep("chargesClustered2",names(training.set1))
training.set1<-training.set1[,c(col_idx2,(1:ncol(training.set1))[-col_idx2])]

col_idx3<-grep("chargesClustered3",names(training.set1))
training.set1<-training.set1[,c(col_idx3,(1:ncol(training.set1))[-col_idx3])]

col_idx4<-grep("patient_id",names(training.set1))
training.set1<-training.set1[,c(col_idx4,(1:ncol(training.set1))[-col_idx4])]

training.set<-training.set1

#Create 3 datasets for the clusters to build models in

c1 <- training.set[training.set$chargesClustered == 1,]
c2 <- training.set[training.set$chargesClustered == 2,]
c3 <- training.set[training.set$chargesClustered == 3,]

save(training.set, testing.set, validation.set, c1, c2, c3, file="/project/msca/capstone3/datasplits.RData")

#Build SVM
training.setsvm<-training.set
training.setsvm$patient_id<-NULL
svm.model<-svm(chargesClustered1~., data=training.setsvm)
svm.model2<-svm(chargesClustered2~.,data=training.setsvm)
svm.model3<-svm(chargesClustered3~.,data=training.setsvm)


bootControl<-trainControl(number=200)
set.seed(10)
svm.model1<-train(chargesClustered1~., data=training.setsvm, method="svmRadial", tuneLength=5, trControl=bootControl, scaled=FALSE)
svm.model1$finalModel

svm.model2<-train(chargesClustered2~., data=training.setsvm, method="svmRadial", tuneLength=5, trControl=bootControl, scaled=FALSE)
svm.model2$finalModel


totalcharges<-sum(training.set$y2_charges)
lowsvm<-subset(training.set, y2_charges<15000)
66165733/356577371 #low subset represents 18.6% of the total charges
totalchargeslow<-sum(lowsvm$y2_charges)
highsvm<-subset(training.set, y2_charges>=15000)
highsvm$high<-1
col_idx5<-grep("high",names(highsvm))
highsvm<-highsvm[,c(col_idx5,(1:ncol(highsvm))[-col_idx5])]




svmmulti<-ksvm(chargesClustered~., data=training.setsvm, scaled=TRUE, type="spoc-svc", kernel="rbfdot")
svmmulti
