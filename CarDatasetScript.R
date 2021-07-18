### MULTIPLE LINEAR REGRESSION ON CAR DTASET ###
car=read.csv("C:/Users/vaish/Downloads/Car.csv", header =T,na.strings ="?")

dim(car)
car [1:4 ,]

#Fitting the model to all the predictors 
multiple_lm.fit =lm(price~.,data=car)
#gives summary statistics of the fit instance
summary(multiple_lm.fit)

#function to calculate mean square error
calculate_mse = function(model,df, predictions, target){
  resids = df[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  print("RMSE:")
  print(as.character(round(sqrt(sum(resids2)/N), 2))) #RMSE
  print("Adjusted R-squared:")
  adj_r2 = as.character(round(summary(model)$adj.r.squared, 2))
  print(adj_r2)
  print("R squared:")
  r2 = as.character(round(summary(model)$r.squared, 4))
  print(r2)
}

set.seed(100) # to get same random division each time

index <- sample(nrow(car), nrow(car) * 0.70)
car.train <- car[index, ]
car.test <- car[-index, ]

#cols = c(colnames(car[-22])) #removing the target column (price)

pre_proc_val <- preProcess(car.train[,cols], method = c("center", "scale"))

car.train[,cols] = predict(pre_proc_val, car.train[,cols])
car.test[,cols] = predict(pre_proc_val, car.test[,cols])

#plot to visualize linearity with target variable
linearity_fit1=lm(price~lstat,data=car)
plot(price~lstat,car)
abline(linearity_fit1,col="red")

linearity_fit2=lm(price~lstat +I(lstat^2), car)
attach(car)
par(mfrow=c(1,1))
plot(price~lstat, car)
points(lstat,fitted(linearity_fit2), col="red",pch=20)

carCor <- cor(car)
#print(car.cor)
library(corrplot)
corrplot(carCor, method="circle")

#new lm object 
fit8=lm(medv~lstat+crim+rm+dis+black+chas+nox+rad+tax+ptratio+I(lstat^2), data = car.train)

predictions_train = predict(fit8, newdata = car.train)
print("For Car Training Data: ")
calculate_mse(fit8, car.train, predictions_train, target = 'price')

predictions_test = predict(fit8, newdata = car.test)
print("For Car Test Data: ")
calculate_mse(fit8, car.test, predictions_test, target = 'price')

n = 1:length(car.test$price)
plot(n, car.test$price, pch=18, col="red")
lines(n, predictions_test, lwd="1", col="blue")

### RIDGE REGRESSION ON CAR DATASET ###
car=read.csv("C:/Users/vaish/Downloads/Car.csv", header =T,na.strings ="?")
fix(car)
dim(car)
car [1:4 ,]

library(tidyverse)
library(glmnet)
library(caret)

#set a seed so you can reproduce the results
set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(car))
training_index <- sample(seq_len(nrow(car)), size = sample_size)
train <- car[training_index, ]
test <- car[-training_index, ]

# Predictor
x <- model.matrix(price~., train)[,-1]
# Response
y <- train$price

cv.r <- cv.glmnet(x, y, alpha = 0.5)
cv.r$lambda.min
model.ridge <- glmnet(x, y, alpha = 0.5, lambda = cv.r$lambda.min)
coef(model.ridge)

x.train.ridge <- model.matrix(price ~., train)[,-1]
predictions.ridge <- model.ridge %>%
  predict(x.train.ridge) %>%
  as.vector()

print("RIDGE REGRESSION")
print("Car Dataset Training:")
data.frame(
  RMSE.r = RMSE(predictions.ridge, train$price),
  Rsquare.r = R2(predictions.ridge, train$price))

x.test.ridge <- model.matrix(price ~., test)[,-1]
predictions.ridge <- model.ridge %>%
  predict(x.test.ridge) %>%
  as.vector()

print("Car Dataset Test:")
data.frame(
  RMSE.r = RMSE(predictions.ridge, test$price),
  Rsquare.r = R2(predictions.ridge, test$price))

### LASSO REGULARIZATION ON CAR DATASET ###
car=read.csv("C:/Users/vaish/Downloads/Car.csv", header =T,na.strings ="?")
fix(car)
dim(car)
car [1:4 ,]

library(tidyverse)
library(glmnet)
library(caret)

#set a seed so you can reproduce the results
set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(car))
training_index <- sample(seq_len(nrow(car)), size = sample_size)
train <- car[training_index, ]
test <- car[-training_index, ]

# Predictor
x <- model.matrix(price~., train)[,-1]
# Response
y <- train$price

#LASSO
cv.l <- cv.glmnet(x, y, alpha = 1)
cv.l$lambda.min
model.lasso <- glmnet(x, y, alpha = 1, lambda = cv.l$lambda.min)

coef(model.lasso)
x.train.lasso <- model.matrix(price ~., train)[,-1]
predictions.lasso <- model.lasso %>%
  predict(x.train.lasso) %>% 
  as.vector()

print("LASSO")

print("Car Dataset Training:")
data.frame(
  RMSE.l = RMSE(predictions.lasso, train$price),
  Rsquare.l = R2(predictions.lasso, train$price))

coef(model.lasso)
x.test.lasso <- model.matrix(price ~., test)[,-1]
predictions.lasso <- model.lasso %>%
  predict(x.test.lasso) %>% 
  as.vector()
print("Car Dataset Testing:")
data.frame(
  RMSE.l = RMSE(predictions.lasso, test$price),
  Rsquare.l = R2(predictions.lasso, test$price))


### ELASTIC NET REGULARIZATION ON CAR DATASET ###
install.packages("corrplot")

source("http://www.sthda.com/upload/rquery_cormat.r")

car=read.csv("C:/Users/vaish/Downloads/Car.csv", header =T,na.strings ="?")
fix(car)
dim(car)
car [1:4 ,]

mydata.cor = cor(car)
print(mydata.cor)

library(tidyverse)
library(glmnet)
library(caret)

#set a seed so you can reproduce the results
set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(car))
training_index <- sample(seq_len(nrow(car)), size = sample_size)
train <- car[training_index, ]
test <- car[-training_index, ]

# Predictor
x <- model.matrix(price~., train)[,-1]
# Response
y <- train$price

#ELASTIC NET
model.net <- train(
  price ~., data = train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10)
model.net$bestTune

coef(model.net$finalModel, model.net$bestTune$lambda)
x.train.net <- model.matrix(price ~., train)[,-1]
predictions.net <- model.net %>% predict(x.train.net)

print("ELASTIC NET")

print("Car Dataset Training:")
data.frame(
  RMSE.net = RMSE(predictions.net, train$price),
  Rsquare.net = R2(predictions.net, train$price))


coef(model.net$finalModel, model.net$bestTune$lambda)
x.test.net <- model.matrix(price ~., test)[,-1]
predictions.net <- model.net %>% predict(x.test.net)

print("Car Dataset Test:")
data.frame(
  RMSE.net = RMSE(predictions.net, test$price),
  Rsquare.net = R2(predictions.net, test$price))


### SVM METHOD ON CAR DATASET ###
install.packages("e1071", dependencies=TRUE)
install.packages("caret", dependencies=TRUE)

library(e1071)
library(caret)

car=read.csv("C:/Users/vaish/Downloads/Car.csv", header =T,na.strings ="?")
dim(car)
car [1:4 ,]

#set a seed so you can reproduce the results
set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(car))
training_index <- sample(seq_len(nrow(car)), size = sample_size)
train <- car[training_index, ]
test <- car[-training_index, ]

model_reg = svm(price~., data=train)
print(model_reg)

pred_train = predict(model_reg, train)
pred_test = predict(model_reg, test)

x = 1:length(test$price)
plot(x, test$price, pch=18, col="red")
lines(x, pred, lwd="1", col="blue")

print("SVM for Car Price train dataset")
length(train$price)
length(pred)
rmse_train = RMSE(train$price, pred_train)
r2_train = R2(train$price, pred_train, form = "traditional")
cat("RMSE:", rmse_train, "\n", "R-squared:", r2_train)

print("SVM for Car Price test dataset")
rmse_test = RMSE(test$price, pred_test) 
r2_test = R2(test$price, pred_test, form = "traditional")
cat("RMSE:", rmse_test, "\n", "R-squared:", r2_test)


### RIDGE REGRESSION REGULARIZATION ON SVM USING BOSTON DATASET ###
car=read.csv("C:/Users/vaish/Downloads/Car.csv", header =T,na.strings ="?")
dim(car)
car [1:4 ,]

#function to calculate mean square error
calculate_metric = function(car, predictions, target){
  resids = car[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  print(paste0("RMSE: ",as.character(round(sqrt(sum(resids2)/N), 2)))) #RMSE
}

set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(car))
training_index <- sample(seq_len(nrow(car)), size = sample_size)
train <- car[training_index, ]
test <- car[-training_index, ]

tune.out=tune(svm , price~., data=train, kernel ="radial",ranges =list(epsilon=seq(0,1,0.1),cost=c(0.1 ,1 ,10 ,100), gamma=c(0.5,1,2,3,4)))
summary (tune.out)


svmfit =svm(price~., data=train , kernel ="radial", epsilon =0.2, cost =10, gamma =0.5, scale =TRUE)

predictions_train <- predict(svmfit, train)
calculate_metric(train, predictions_train, target = 'price')

predictions_test <- predict(svmfit, test)
calculate_metric(test, predictions_test, target = 'price')

n = 1:length(test$price)
plot(n, test$medv, pch=18, col="red",ylab = "Data Points",xlab = "Index")
title("SVM - Test and Predicted data (Log Transformed)")
lines(n, predictions_test, lwd="1", col="blue")
legend("topleft", legend=c("Test","Predicted"),col=c("red","blue"),pch = 16:16, cex=0.6)

#plot to visualize how well the model follows the logged median value
n = 1:length(exp(test$price))
plot(n, exp(test$price), pch=18, col="red",ylab = "Data Points",xlab = "Index")
title("SVM - Actual and Predicted data (Actual)")
lines(n, exp(predictions_test), lwd="1", col="blue")
legend("topleft", legend=c("Actual","Predicted"),col=c("red","blue"),pch = 16:16, cex=0.6)

#Error w.r.t median values
train_error = sqrt(mean((exp(train$price)-exp(predictions_train))^2))
test_error = sqrt(mean((exp(test$price)-exp(predictions_test))^2))
print(paste0("RMSE of Actual Train Set: ", round(train_error, 2)))
print(paste0("RMSE of Actual Test Set: ", round(test_error, 2)))

### LASSO AND ELASTIC NET REGULARIZATION ON SVM USING BOSTON DATASET ###
#to supress warning
options(warn=-1)

#installing packages and adding to library
#install.packages("sparseSVM", dependencies = TRUE)
library(sparseSVM)
library(MASS)
library(caret)

car=read.csv("C:/Users/vaish/Downloads/Car.csv", header =T,na.strings ="?")
dim(car)
car [1:4 ,]

#function to calculate mean square error
calculate_metric = function(boston, predictions, target){
  resids = boston[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  print(paste0("RMSE: ",as.character(round(sqrt(sum(resids2)/N), 2)))) #RMSE
}

set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(car))
training_index <- sample(seq_len(nrow(car)), size = sample_size)
train <- car[training_index, ]
test <- car[-training_index, ]

cols = c(colnames(car[-1]))

pre_proc_val <- preProcess(train[,cols], method = c("center", "scale"))

train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])

cols_reg = c(colnames(car))

#The lines of code below perform the task of creating model matrix using the 
#dummyVars function from the caret package
dummies <- dummyVars(price ~ ., data = boston[,cols_reg])

train_dummies = predict(dummies, newdata = train[,cols_reg])

test_dummies = predict(dummies, newdata = test[,cols_reg])


#creating training and test matrices for independent and dependent variables
x = as.matrix(train_dummies)
y_train = train$price

x_test = as.matrix(test_dummies)
y_test = test$price


yclass_train = ifelse(y_train>10, 1, 0)
yclass_test = ifelse(y_test>10, 1, 0)

#sparseSVM using lasso regularization
cv.fit <- cv.sparseSVM(x, yclass_train, alpha = 1, ncores = 2, eval.metric = c("me"),
                       nfolds = 10, trace = FALSE)

predictedY <- predict(cv.fit, x)
a <- as.factor(predictedY)
b <- as.factor(yclass_train)
cm_train <- confusionMatrix(a, b)
predictedY <- predict(cv.fit, x_test)
a <- as.factor(predictedY)
b <- as.factor(yclass_test)
cm_test <- confusionMatrix(a, b)
print("SVM with Lasso")
print(paste0("Lasso - Train Accuracy :",round(cm_train$overall['Accuracy'],2)*100))
print(paste0("Lasso - Test Accuracy :",round(cm_test$overall['Accuracy'],2)*100))

#sparseSVM using elastic net regularization
cv.fit <- cv.sparseSVM(x, yclass_train, alpha = 0.8, ncores = 2, eval.metric = c("me"),
                       nfolds = 10, trace = FALSE)

predictedY <- predict(cv.fit, x)
a <- as.factor(predictedY)
b <- as.factor(yclass_train)
cm_train <- confusionMatrix(a, b)
predictedY <- predict(cv.fit, x_test)
a <- as.factor(predictedY)
b <- as.factor(yclass_test)
cm_test <- confusionMatrix(a, b)
print("SVM using Elastic Net")
print(paste0("Elastic Net - Train Accuracy :",round(cm_train$overall['Accuracy'],2)*100))
print(paste0("Elastic Net - Test Accuracy :",round(cm_test$overall['Accuracy'],2)*100))