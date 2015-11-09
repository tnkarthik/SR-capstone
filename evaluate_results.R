results_oob_rf<-evaluate_results(lib_train)
results_oob_rf_sampled<-evaluate_results(data_sampled)
results_oob_rf_sampled_inc <- evaluate_results(data_sampled_inc)
results_oob_rf_smote2 <-  evaluate_results(data_smote2)

results_oob <- rbind(results_oob_rf, results_oob_rf_sampled, results_oob_rf_sampled_inc,
                     results_oob_rf_smote2)
results_oob<-add_rownames(data.frame(results_oob))
results_oob_long <- melt(results_oob, id.vars = "rowname" )

evaluate_results <- function(data_set) {
  
  library(dplyr)
  
  temp_low <- filter(data_set, H_cat == "low")
  temp_medium <- filter(data_set, H_cat == "medium")
  temp_high <- filter(data_set, H_cat == "high")
  
 result<-(c(RMSE(temp_low$pred,temp_low$Hazard),
                RMSE(temp_medium$pred,temp_medium$Hazard),
                RMSE(temp_high$pred,temp_high$Hazard)))
  names(result)<- c("low","medium","high")
  return(result)
}

p_testsetresults_oob <- ggplot(aes(x=variable,y=value,fill=rowname),data=results_oob_long)+
  geom_bar(stat="identity",position="dodge")+
  ggtitle("OOB Error Results")+
  xlab("Hazard Category")+
  ylab("RMSE")+
  scale_fill_manual(values=c("darkred", "blue", "green" , "orange"), 
                    name="Model",
                    
                    labels=c("w/o sampling", "undersampling", "oversampling" , "SMOTE"))+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20),
        axis.title = element_text(lineheight=.8, face="bold",size=15),
        axis.text = element_text(lineheight=.8, face="bold",size=14))

jpeg("Results.jpeg",height=10 , width = 10 , units= 'in' , res=300)
grid.arrange(p_testsetresults_oob,p_testsetresults,nrow=2)
dev.off()

p_final <- grid.arrange(p_testsetresults_oob,p_testsetresults,nrow=2)
