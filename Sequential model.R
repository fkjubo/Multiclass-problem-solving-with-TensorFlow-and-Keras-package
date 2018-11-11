library(keras)

# importing data

data.train <- read.csv("train (1).csv", T, ",")
data.test <- read.csv("test (1).csv", T, ",")

# changing data into matrix

data.train <- as.matrix(data.train)
data.test <- as.matrix(data.test)

# deleting names of the data

dimnames(data.train) <- NULL
dimnames(data.test) <- NULL

# normalise the data

data.trainN <- data.train
data.testN <- data.test
data.trainN <- normalize(data.trainN[,1:21])
data.testN <- normalize(data.testN)

data.trainN <- data.trainN[, 1:21]
data.trainNTarget <- data.train[,22]
data.trainNTarget <- as.numeric(data.trainNTarget)-1
# we didn't create data.trainNTarget because our test data hasn't got NSP variable
# as it is kaggle competition

# one hot encoding
# if the data is only 0 and 1 then we don't need to do this step

trainLabels <- to_categorical(data.trainNTarget)

# sequential model

model <- keras_model_sequential() 

model %>%
  layer_dense(units = 6, activation = "relu", input_shape = c(21)) %>%
  layer_dense(units = 3, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")

# compiling the model

model %>%
  compile(loss = "categorical_crossentropy",
          optimizer = "adam",
          metrics = "accuracy")

# fitting the model

history <- model %>%
  fit(data.trainN,
      trainLabels,
      validation_split = .2,
      batch_size = 32,
      epochs = 350)

# predicting the model

pred <- model %>%
  predict_classes(data.testN)

# showing the probability of the model decison

prob <- model %>%
  predict_proba(data.testN)

View(pred)
str(pred)

# adding +1 to reflect the NSP data which we subtracted for the model

NSP <- pred + 1

NSP <- as.data.frame(NSP)

write.csv(pred, "kaggle.csv")
