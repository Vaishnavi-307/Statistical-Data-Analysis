### MULTILINEAR REGRESSION OF BOSTON DATASET ###

library(MASS)
library(caret)
Boston
dim(Boston)
fix(boston)

#Fitting the model to all the predictors 
multiple_lm.fit =lm(medv~.,data=Boston)
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

bostonCor <- cor(boston)
#print(car.cor)
library(corrplot)
corrplot(bostonCor, method="circle")

set.seed(100) # to get same random division each time

index <- sample(nrow(Boston), nrow(Boston) * 0.80)
Boston.train <- Boston[index, ]
Boston.test <- Boston[-index, ]

fit <- multiple_lm.fit 

predictions_train = predict(fit, newdata = Boston.train)
print("For Boston Training Data: ")
calculate_mse(fit, Boston.train, predictions_train, target = 'medv')

predictions_test = predict(fit, newdata = Boston.test)
print("For Boston Test Data: ")
calculate_mse(fit, Boston.test, predictions_test, target = 'medv')

n = 1:length(Boston.test$medv)
plot(n, Boston.test$medv, pch=18, col="red")
lines(n, predictions_test, lwd="1", col="blue")


### RIDGE REGRESSION REGULARIZATION METHOD ON BOSTON DATASET ###
library(MASS)
data(Boston)
head(Boston)

library(tidyverse)
library(glmnet)
library(caret)

#RIDGE REGRESSION
#set a seed so you can reproduce the results
set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(Boston))
training_index <- sample(seq_len(nrow(Boston)), size = sample_size)
train <- Boston[training_index, ]
test <- Boston[-training_index, ]

# Predictor
x <- model.matrix(medv~., train)[,-1]
# Response
y <- train$medv

cv.r <- cv.glmnet(x, y, alpha = 0.5)
cv.r$lambda.min
model.ridge <- glmnet(x, y, alpha = 0.5, lambda = cv.r$lambda.min)
coef(model.ridge)

x.train.ridge <- model.matrix(medv ~., train)[,-1]
predictions.ridge <- model.ridge %>%
  predict(x.train.ridge) %>%
  as.vector()
print("Boston dataset training: Ridge Regression")
data.frame(
  RMSE.r = RMSE(predictions.ridge, train$medv),
  Rsquare.r = R2(predictions.ridge, train$medv))

x.test.ridge <- model.matrix(medv ~., test)[,-1]
predictions.ridge <- model.ridge %>%
  predict(x.test.ridge) %>%
  as.vector()
print("Boston dataset testing: Ridge Regression")
data.frame(
  RMSE.r = RMSE(predictions.ridge, test$medv),
  Rsquare.r = R2(predictions.ridge, test$medv))


### LASSO REGULARIZATION METHOD ON BOSTON DATASET ###
library(MASS)
data(Boston)
head(Boston)

#set a seed so you can reproduce the results
set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(Boston))
training_index <- sample(seq_len(nrow(Boston)), size = sample_size)
train <- Boston[training_index, ]
test <- Boston[-training_index, ]

# Predictor
x <- model.matrix(medv~., train)[,-1]
# Response
y <- train$medv

#LASSO
cv.l <- cv.glmnet(x, y, alpha = 1)
cv.l$lambda.min
model.lasso <- glmnet(x, y, alpha = 1, lambda = cv.l$lambda.min)

coef(model.lasso)
x.train.lasso <- model.matrix(medv ~., train)[,-1]
predictions.lasso <- model.lasso %>%
  predict(x.train.lasso) %>% 
  as.vector()

data.frame(
  RMSE.l = RMSE(predictions.lasso, train$medv),
  Rsquare.l = R2(predictions.lasso, train$medv))

coef(model.lasso)
x.test.lasso <- model.matrix(medv ~., test)[,-1]
predictions.lasso <- model.lasso %>%
  predict(x.test.lasso) %>% 
  as.vector()

data.frame(
  RMSE.l = RMSE(predictions.lasso, test$medv),
  Rsquare.l = R2(predictions.lasso, test$medv))


### ELASTIC NET REGULARIZATION METHOD ON BOSTON DATASET ###
library(MASS)
data(Boston)
head(Boston)

#set a seed so you can reproduce the results
set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(Boston))
training_index <- sample(seq_len(nrow(Boston)), size = sample_size)
train <- Boston[training_index, ]
test <- Boston[-training_index, ]

# Predictor
x <- model.matrix(medv~., train)[,-1]
# Response
y <- train$medv

#ELASTIC NET
model.net <- train(
  medv ~., data = train, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10)
model.net$bestTune

coef(model.net$finalModel, model.net$bestTune$lambda)
x.train.net <- model.matrix(medv ~., train)[,-1]
predictions.net <- model.net %>% predict(x.train.net)

data.frame(
  RMSE.net = RMSE(predictions.net, train$medv),
  Rsquare.net = R2(predictions.net, train$medv))


coef(model.net$finalModel, model.net$bestTune$lambda)
x.test.net <- model.matrix(medv ~., test)[,-1]
predictions.net <- model.net %>% predict(x.test.net)

data.frame(
  RMSE.net = RMSE(predictions.net, test$medv),
  Rsquare.net = R2(predictions.net, test$medv))

### SVM METHOD ON BOSTON DATASET ###
library(e1071)
library(caret)

boston = MASS::Boston
#set a seed so you can reproduce the results
set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(boston))
training_index <- sample(seq_len(nrow(boston)), size = sample_size)
train <- boston[training_index, ]
test <- boston[-training_index, ]

model_reg = svm(medv~., data=train)
print(model_reg)

pred_train = predict(model_reg, train)
pred_test = predict(model_reg, test)

x = 1:length(test$medv)
plot(x, test$medv, pch=18, col="red")
lines(x, pred, lwd="1", col="blue")

print("SVM for Boston train dataset")
rmse_train = RMSE(train$medv, pred_train)
r2_train = R2(train$medv, pred_train, form = "traditional")
cat("RMSE:", rmse_train, "\n", "R-squared:", r2_train)

print("SVM for Boston test dataset")
rmse_test = RMSE(test$medv, pred_test)
r2_test = R2(test$medv, pred_test, form = "traditional")
cat("RMSE:", rmse_test, "\n", "R-squared:", r2_test)
length(pred_test)
length(test$medv)

### RIDGE REGRESSION REGULARIZATION ON SVM USING BOSTON DATASET ###
boston<- Boston
print(boston)

#function to calculate mean square error
calculate_metric = function(boston, predictions, target){
  resids = boston[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  print(paste0("RMSE: ",as.character(round(sqrt(sum(resids2)/N), 2)))) #RMSE
}

set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(boston))
training_index <- sample(seq_len(nrow(boston)), size = sample_size)
train <- boston[training_index, ]
test <- boston[-training_index, ]

tune.out=tune(svm , medv~crim+zn+indus+chas+nox+rm+age+dis+rad+ptratio+black+lstat, data=train, kernel ="radial",ranges =list(epsilon=seq(0,1,0.1),cost=c(0.1 ,1 ,10 ,100), gamma=c(0.5,1,2,3,4)))
summary (tune.out)


svmfit =svm(medv~crim+zn+indus+chas+nox+rm+age+dis+rad+ptratio+black+lstat, data=train , kernel ="radial", epsilon =0.2, cost =10, gamma =0.5, scale =TRUE)

predictions_train <- predict(svmfit, train)
calculate_metric(train, predictions_train, target = 'medv')

predictions_test <- predict(svmfit, test)
calculate_metric(test, predictions_test, target = 'medv')

n = 1:length(test$medv)
plot(n, test$medv, pch=18, col="red",ylab = "Data Points",xlab = "Index")
title("SVM - Test and Predicted data (Log Transformed)")
lines(n, predictions_test, lwd="1", col="blue")
legend("topleft", legend=c("Test","Predicted"),col=c("red","blue"),pch = 16:16, cex=0.6)

#plot to visualize how well the model follows the logged median value
n = 1:length(exp(test$medv))
plot(n, exp(test$medv), pch=18, col="red",ylab = "Data Points",xlab = "Index")
title("SVM - Actual and Predicted data (Actual)")
lines(n, exp(predictions_test), lwd="1", col="blue")
legend("topleft", legend=c("Actual","Predicted"),col=c("red","blue"),pch = 16:16, cex=0.6)

#Error w.r.t median values
train_error = sqrt(mean((exp(train$medv)-exp(predictions_train))^2))
test_error = sqrt(mean((exp(test$medv)-exp(predictions_test))^2))
print(paste0("RMSE of Actual Train Set: ", round(train_error, 2)))
print(paste0("RMSE of Actual Test Set: ", round(test_error, 2)))

### LASSO AND ELASTIC NET REGULARIZATION ON SVM USING BOSTON DATASET ###
#to supress warning
options(warn=-1)

#installing packages and adding to library
#install.packages("sparseSVM", depencies = TRUE)
library(sparseSVM)
library(MASS)
library(caret)

boston<- Boston
print(boston)

#function to calculate mean square error
calculate_metric = function(boston, predictions, target){
  resids = boston[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  print(paste0("RMSE: ",as.character(round(sqrt(sum(resids2)/N), 2)))) #RMSE
}

set.seed(100)
#split the data into training and test data
sample_size <- floor(0.80 * nrow(boston))
training_index <- sample(seq_len(nrow(boston)), size = sample_size)
train <- boston[training_index, ]
test <- boston[-training_index, ]

cols = c(colnames(boston[-1]))

pre_proc_val <- preProcess(train[,cols], method = c("center", "scale"))

train[,cols] = predict(pre_proc_val, train[,cols])
test[,cols] = predict(pre_proc_val, test[,cols])

cols_reg = c(colnames(boston))

#The lines of code below perform the task of creating model matrix using the 
#dummyVars function from the caret package
dummies <- dummyVars(crim ~ ., data = boston[,cols_reg])

train_dummies = predict(dummies, newdata = train[,cols_reg])

test_dummies = predict(dummies, newdata = test[,cols_reg])


#creating training and test matrices for independent and dependent variables
x = as.matrix(train_dummies)
y_train = train$medv

x_test = as.matrix(test_dummies)
y_test = test$medv


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
print("svm with lasso")
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
print("svm with elastic net")
print(paste0("Elastic Net - Train Accuracy :",round(cm_train$overall['Accuracy'],2)*100))
print(paste0("Elastic Net - Test Accuracy :",round(cm_test$overall['Accuracy'],2)*100))
