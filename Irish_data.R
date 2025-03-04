library(readxl)
library(neuralnet)
library(Metrics)
library(e1071)
library(tseries)
library(openxlsx)

data <- read_excel('Irish_data.xlsx')

set.seed(01112001)

train_unscaled <- data[1:14,]
test_unscaled <- data[15:19,]

train_scaled <- as.data.frame(scale(train_unscaled))
train_mean <- sapply(train_unscaled, mean)
train_sd <- sapply(train_unscaled, sd)
test_scaled <- as.data.frame(scale(test_unscaled, center=train_mean, 
                                   scale=train_sd))
#looking for correlation
cor(data[,2:7])

#looking for trend
plot(x=data$year, y=data$MW, type='b' ,xlab='Year', ylab='Waste (kg/capita)', 
     xaxt = "n", cex.lab=1.2)
axis(1, at = data$year, labels = FALSE, tcl = -0.5) #tick marks for every year
text(
  x=data$year[data$year%%2==0],      #positions for labels
  y=540,                             #slightly below the axis
  labels=data$year[data$year%%2==0], #adding the labels
  srt=45,                            #rotation angle in degrees
  adj=0.5,                           #alignment (0.5=centered)
  xpd=TRUE,                          #allow drawing outside the plot area
  cex=0.9                            #size of the text
)

adf.test(data$MW)

#fitting a neural network to training data ----
set.seed(01112001)
network <- neuralnet(MW~GEC+GDP+population+unemployment+C02+inflation, 
                     data=train_scaled, hidden=c(10), linear.output=TRUE)
predictions <- predict(network, train_scaled)
actual <- train_scaled$MW

predictions_unscaled <- predictions*sd(train_unscaled$MW)+mean(train_unscaled$MW)
residuals <- train_unscaled$MW-predictions_unscaled
plot(predictions_unscaled, residuals, xlab='Fitted values (kg/capita)', 
     ylab='Residuals (kg/capita)', xaxt="n", yaxt="n", cex.lab=1.2) #exhibits heteroscadicity
axis(1, at = seq(550, 800, by = 25))
axis(2, at = seq(-4, 4, by = 0.5))

# Compute R^2
SSE <- sum((train_unscaled$MW-predictions_unscaled)^2)
SST <- sum((train_unscaled$MW-mean(train_unscaled$MW))^2)
R2 <- 1-SSE/SST
cat("R^2:", R2, "\n") #very high

rmse(train_unscaled$MW,predictions_unscaled)
mae(train_unscaled$MW,predictions_unscaled)

plot(train_unscaled$MW, type='l')
lines(predictions_unscaled, col='red')

#now let's look at the test set----
predictions_test <- predict(network, test_scaled)
actual_test <- test_scaled$MW

predictions_unscaled_test <- predictions_test*sd(train_unscaled$MW)+mean(train_unscaled$MW)
residuals_test <- test_unscaled$MW-predictions_unscaled_test
plot(predictions_unscaled_test, residuals_test, xlab='Fitted', ylab='Residuals', 
     main = 'Fitted vs Residuals - Test')

# Compute R^2
SSE <- sum((test_unscaled$MW-predictions_unscaled_test)^2)
SST <- sum((test_unscaled$MW-mean(test_unscaled$MW))^2)
R2 <- 1-SSE/SST
cat("R^2:", R2, "\n") #again very high

rmse(test_unscaled$MW,predictions_unscaled_test)
mae(test_unscaled$MW,predictions_unscaled_test)

plot(test_unscaled$MW, type='l')
lines(predictions_unscaled_test, col='red')

#is there a cross term?
sum((test_unscaled$MW-mean(test_unscaled$MW))^2) #sst
sum((predictions_unscaled_test-mean(test_unscaled$MW))^2)+
  sum((test_unscaled$MW-predictions_unscaled_test)^2) #SSR+SSE
#cross term
2*sum((predictions_unscaled_test-mean(test_unscaled$MW))*(test_unscaled$MW-predictions_unscaled_test))

#the above two aren't the same (that is SST doesn't equal SSE+SSR), indicating 
#that the residuals haven't been partitoned correctly

# lets try an svm model----
svm_model <- svm(MW~GEC+GDP+population+unemployment+C02+inflation, 
                 data=train_scaled, type='eps-regression', kernal='radial')

predictions_svm <- predict(svm_model, train_scaled)

predictions_unscaled_svm <- predictions_svm*sd(train_unscaled$MW)+mean(train_unscaled$MW)
residuals_svm <- train_unscaled$MW-predictions_unscaled_svm
plot(predictions_unscaled_svm, residuals_svm, xlab='Fitted values (kg/capita)', 
     ylab='Residuals (kg/capita)', xaxt="n", yaxt="n", cex.lab=1.2)
axis(1, at = seq(575, 800, by = 25))
axis(2, at = seq(-60, 40, by = 20))

# Compute R^2
SSE <- sum((train_unscaled$MW-predictions_unscaled_svm)^2)
SST <- sum((train_unscaled$MW-mean(train_unscaled$MW))^2)
R2 <- 1-SSE/SST
cat("R^2:", R2, "\n") #very high

rmse(train_unscaled$MW, predictions_unscaled_svm)
mae(train_unscaled$MW, predictions_unscaled_svm)

plot(train_unscaled$MW, type='l')
lines(predictions_unscaled_svm, col='red')

#test set
predictions_test_svm <- predict(svm_model, test_scaled)
actual_test <- test_scaled$MW

predictions_unscaled_test_svm <- predictions_test_svm*sd(train_unscaled$MW)+mean(train_unscaled$MW)
residuals_test_svm <- test_unscaled$MW-predictions_unscaled_test_svm
plot(predictions_unscaled_test_svm, residuals_test_svm, xlab='Fitted', 
     ylab='Residuals', main = 'Fitted vs Residuals - Test (svm)')

# Compute R^2
SSE <- sum((test_unscaled$MW-predictions_unscaled_test_svm)^2)
SST <- sum((test_unscaled$MW-mean(test_unscaled$MW))^2)
R2 <- 1-SSE/SST
cat("R^2:", R2, "\n")

rmse(test_unscaled$MW,predictions_unscaled_test_svm)
mae(test_unscaled$MW,predictions_unscaled_test_svm)

plot(test_unscaled$MW, type='l')
lines(predictions_unscaled_test_svm, col='red')

#is there a cross term?
sum((test_unscaled$MW-mean(test_unscaled$MW))^2) #sst
sum((predictions_unscaled_test_svm-mean(test_unscaled$MW))^2)+
  sum((test_unscaled$MW-predictions_unscaled_test_svm)^2) #SSR+SSE

#the above two aren't the same (that is SST doesn't equal SSE+SSR), indicating 
#that the residuals haven't been partitioned correctly

#cross term
2*sum((predictions_unscaled_test_svm-mean(test_unscaled$MW))*(test_unscaled$MW-predictions_unscaled_test_svm))



#getting the data that's used in plots in the manuscript ----
#the dependent plot 
figure_2 <- data.frame(year=data$year, waste=data$MW)
write.xlsx(figure_2, "figure_2.xlsx")

#neural network residual vs fitted
figure_3 <- data.frame(fitted=predictions_unscaled, residuals=residuals)
write.xlsx(figure_3, "figure_3.xlsx")

#svm residuals vs fitted
figure_4 <- data.frame(fitted=predictions_unscaled_svm, residuals=residuals_svm)
write.xlsx(figure_4, "figure_4.xlsx")



#network with 5 neruons ----
set.seed(01112001)
network <- neuralnet(MW~GEC+GDP+population+unemployment+C02+inflation, 
                     data=train_scaled, hidden=c(5), linear.output=TRUE)
predictions <- predict(network, train_scaled)
actual <- train_scaled$MW

predictions_unscaled <- predictions*sd(train_unscaled$MW)+mean(train_unscaled$MW)
residuals <- train_unscaled$MW-predictions_unscaled
plot(predictions_unscaled, residuals, xlab='Fitted values', ylab='Residuals', 
     xaxt="n", yaxt="n") #exhibits heteroscadicity
axis(1, at = seq(550, 800, by = 25))
axis(2, at = seq(-10, 15, by = 2))

# Compute R^2
SSE <- sum((train_unscaled$MW-predictions_unscaled)^2)
SST <- sum((train_unscaled$MW-mean(train_unscaled$MW))^2)
R2 <- 1-SSE/SST
cat("R^2:", R2, "\n") #very high

rmse(train_unscaled$MW,predictions_unscaled)
mae(train_unscaled$MW,predictions_unscaled)

plot(train_unscaled$MW, type='l')
lines(predictions_unscaled, col='red')

#now let's look at the test set----
predictions_test <- predict(network, test_scaled)
actual_test <- test_scaled$MW

predictions_unscaled_test <- predictions_test*sd(train_unscaled$MW)+mean(train_unscaled$MW)
residuals_test <- test_unscaled$MW-predictions_unscaled_test
plot(predictions_unscaled_test, residuals_test, xlab='Fitted', ylab='Residuals', 
     main = 'Fitted vs Residuals - Test')

# Compute R^2
SSE <- sum((test_unscaled$MW-predictions_unscaled_test)^2)
SST <- sum((test_unscaled$MW-mean(test_unscaled$MW))^2)
R2 <- 1-SSE/SST
cat("R^2:", R2, "\n") #again very high

rmse(test_unscaled$MW,predictions_unscaled_test)
mae(test_unscaled$MW,predictions_unscaled_test)

plot(test_unscaled$MW, type='l')
lines(predictions_unscaled_test, col='red')

#is there a cross term?
sum((test_unscaled$MW-mean(test_unscaled$MW))^2) #sst
sum((predictions_unscaled_test-mean(test_unscaled$MW))^2)+
  sum((test_unscaled$MW-predictions_unscaled_test)^2) #SSR+SSE

#cross term
2*sum((predictions_unscaled_test-mean(test_unscaled$MW))*(test_unscaled$MW-predictions_unscaled_test))

