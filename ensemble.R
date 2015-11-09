###ENSEMBLE#####
################


set.seed(1141)
data_smote2_ensemble <- select(data_smote, -rf_sampled , -pred)
breaks_2 <- c(0,20,70)

data_smote_ensemble <- data_smote
data_smote_ensemble$H_cat2<-cut(data_smote_ensemble$Hazard,breaks_2,labels=c("low","med+high"))
lib_test_validation$H_cat2<-cut(lib_test_validation$Hazard,breaks_2,labels=c("low","med+high"))


model_med_high <- train(Hazard~., data = select(filter(data_smote_ensemble,H_cat2 == "med+high"), -Id, -H_cat,-H_cat2),
                      method="rf", tuneGrid = rfgrid,
                      trControl = trainControl(method = "oob"))

model_low2 <- train(Hazard~., data = select(filter(data_smote_ensemble,H_cat2 == "low"), -Id, -H_cat,-H_cat2),
                   method="rf", tuneGrid = rfgrid,
                   trControl = trainControl(method = "oob"))

lib_test_validation$ensemble_cat2 <- predict(model_classify_ensemble_smote1, newdata = lib_test_validation)

low2_pred <- predict(model_low2 , newdata = filter(lib_test_validation, ensemble_cat2 == "low"))
med_high_pred <- predict(model_med_high , newdata = filter(lib_test_validation, ensemble_cat2 == "med+high"))

lib_test_validation <- mutate(lib_test_validation, twostep2 = 0)
lib_test_validation$twostep2[lib_test_validation$ensemble_cat2=="low"]<-low2_pred
lib_test_validation$twostep2[lib_test_validation$ensemble_cat2=="med+high"]<-med_high_pred


temp <- filter(lib_test_validation,H_cat2 == "low")
ensemble_RMSE_low <- RMSE(temp$twostep, 
                          temp$Hazard)

temp <- filter(lib_test_validation,H_cat2 == "med+high")
ensemble_RMSE_medium <- RMSE(temp$twostep, 
                             temp$Hazard)
temp <- filter(lib_test_validation,H_cat == "high")
ensemble_RMSE_high <- RMSE(temp$twostep, 
                           temp$Hazard)


##,-rf_sampled, -pred)
### classification###
rfgrid <- expand.grid(mtry = c(3,5,10,33))

  model_classify_ensemble<-train(H_cat2~.,data = select(data_smote2_ensemble,-Id, -Hazard),
                                 model="rf",tuneGrid = rfgrid,
                                 trControl = trainControl(method = "oob"))
   model_classify_ensemble_smote1<-train(H_cat2~.,data = select(data_smote_ensemble,-Id, H_cat, -Hazard),
                                         model="rf",tuneGrid = expand.grid(mtry = 5),
                                         trControl = trainControl(method = "oob"))
  model_low <- train(Hazard~., data = select(filter(data_smote2_ensemble,H_cat == "low"), -Id, -H_cat),
                     method="rf", tuneGrid = rfgrid,
                     trControl = trainControl(method = "oob"))
  
  model_medium <- train(Hazard~., data = select(filter(data_smote2_ensemble,H_cat == "medium"), -Id, -H_cat),
                     method="rf", tuneGrid = rfgrid,
                     trControl = trainControl(method = "oob"))
  model_high <- train(Hazard~., data = select(filter(data_smote2_ensemble,H_cat == "high"), -Id, -H_cat),
                     method="rf", tuneGrid = rfgrid,
                     trControl = trainControl(method = "oob"))

    lib_test_validation$ensemble_cat <- predict(model_classify_ensemble_smote1, newdata = lib_test_validation)
  
  low_pred <- predict(model_low , newdata = filter(lib_test_validation, ensemble_cat == "low"))
  medium_pred <- predict(model_medium , newdata = filter(lib_test_validation, ensemble_cat == "medium"))
  high_pred <- predict(model_high , newdata = filter(lib_test_validation, ensemble_cat == "high"))
  
  lib_test_validation <- mutate(lib_test_validation, twostep = 0)
  lib_test_validation$twostep[lib_test_validation$ensemble_cat=="low"]<-low_pred
  lib_test_validation$twostep[lib_test_validation$ensemble_cat=="medium"]<-medium_pred
  lib_test_validation$twostep[lib_test_validation$ensemble_cat=="high"]<-high_pred
  
  
  temp <- filter(lib_test_validation,H_cat == "low")
  ensemble_RMSE_low <- RMSE(temp$twostep, 
                            temp$Hazard)
  
  temp <- filter(lib_test_validation,H_cat == "medium")
  ensemble_RMSE_medium <- RMSE(temp$twostep, 
                            temp$Hazard)
  temp <- filter(lib_test_validation,H_cat == "high")
  ensemble_RMSE_high <- RMSE(temp$twostep, 
                            temp$Hazard)
  
  classification_data<-model_classify_ensemble$trainingData
  classification_data<-mutate(classification_data,Hazard=data_sampled_ensemble$Hazard)  
}

