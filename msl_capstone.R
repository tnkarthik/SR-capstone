#####################
##### READ DATA #####
#####################

setwd("C:/Users/Karthik/Documents/GitHub/libmutual/")
lib_train<-read.csv("train.csv")
lib_test<-read.csv("test.csv")

##################################
##### LOAD REQUIRED PACKAGES #####
##################################
library(dplyr)
library(caret)
library(reshape2)
library(ggplot2)

##########################
##### DATA WRANGLING #####
##########################

inTrain<- createDataPartition(y=lib_train$Hazard, p=0.75, list=FALSE)

train_1<-lib_train[inTrain,]
test_1<-lib_train[-inTrain,]

lib_train_all<-lib_train
lib_train<-train_1
lib_test_validation<-test_1
rm(train_1,test_1)

lib_train_long<-melt(select(lib_train,-Id),id.vars = "Hazard",variable.name = "TV")
breaks=c(0,10,20,70)

lib_train$H_cat<-cut(lib_train$Hazard,breaks,labels=c("low","medium","high"))
lib_train_long$H_cat<-cut(lib_train_long$Hazard,breaks,labels=c("low","medium","high"))


#################################
######RANDOM FOREST MODEL########
rfgrid <- expand.grid(mtry = c(2,5,10,33))

set.seed(1141)

t_rf<-proc.time()
    model_rf<-train(Hazard~.,data=select(lib_train,-Id,-H_cat),
                    model="rf",tuneGrid = rfgrid,
                    trControl = trainControl(method = "oob"))
proc.time() - t_rf

print(model_rf)

t_rf_low<-proc.time()
    model_rf_low<-train(Hazard~.,data=select(filter(lib_train,H_cat=="low"),-Id,-H_cat),
                        model="rf",tuneGrid = rfgrid,
                        trControl = trainControl(method = "oob"))
proc.time() - t_rf_low

print(model_rf_low)

t_rf_medium<-proc.time()
    model_rf_medium<-train(Hazard~.,data=select(filter(lib_train,H_cat=="medium"),-Id,-H_cat),
                           model="rf",tuneGrid = rfgrid,
                           trControl = trainControl(method = "oob"))
proc.time() - t_rf_medium

print(model_rf_medium)

t_rf_high<-proc.time()
    model_rf_high<-train(Hazard~.,data=select(filter(lib_train,H_cat=="high"),-Id,-H_cat),
                         model="rf",tuneGrid = rfgrid,
                         trControl = trainControl(method = "oob"))
proc.time() - t_rf_high

print(model_rf_high)


tiff("HazardvsTV_low.tiff",height=30 , width = 30 , units= 'in' , res=300)
p_low<-ggplot(aes(x=value,y=Hazard),data=filter(lib_train_long,H_cat=="low"))+
  geom_boxplot()+
  facet_wrap(~TV,ncol=3, scales = "free_x")
p_low
dev.off()

tiff("HazardvsTV_medium.tiff",height=30 , width = 30 , units= 'in' , res=300)
p_medium<-ggplot(aes(x=value,y=Hazard),data=filter(lib_train_long,H_cat=="medium"))+
  geom_boxplot()+
  facet_wrap(~TV,ncol=3, scales = "free_x")
p_medium
dev.off()

tiff("HazardvsTV_high.tiff",height=30 , width = 30 , units= 'in' , res=300)
p_high<-ggplot(aes(x=value,y=Hazard),data=filter(lib_train_long,H_cat=="high"))+
  geom_boxplot()+
  facet_wrap(~TV,ncol=3, scales = "free_x")
p_high
dev.off()

set.seed(1141)

t_classify<-proc.time()
model_classify<-train(H_cat~.,data=select(lib_train,-Id,-Hazard),
                model="rf",tuneGrid = rfgrid,
                trControl = trainControl(method = "oob"))
proc.time() - t_classify

## sample low and medium entries for better classification

data_sampled_low<-lib_train%>%
  filter(H_cat=="low")%>%
  sample_n(250)

data_sampled_medium<-lib_train%>%
  filter(H_cat=="medium")%>%
  sample_n(250)
data_sampled_high<-lib_train%>%
  filter(H_cat=="high")%>%
  sample_n(250)
data_sampled<-rbind(data_sampled_low,data_sampled_medium,data_sampled_high)
t_classify_sampled<-proc.time()
model_classify_sampled<-train(H_cat~.,data=select(data_sampled,-Id,-Hazard),
                      model="rf",tuneGrid = rfgrid,
                      trControl = trainControl(method = "oob"))
proc.time() - t_classify_sampled

model_classify_sampled$finalModel

## increase medium and high entries for better classification

data_sampled_low<-lib_train%>%
  filter(H_cat=="low")%>%
  sample_n(10000)

data_sampled_medium<-lib_train%>%
  filter(H_cat=="medium")%>%
  sample_n(10000,replace=TRUE)
data_sampled_high<-lib_train%>%
  filter(H_cat=="high")%>%
  sample_n(10000,replace=TRUE)
data_sampled_inc<-rbind(data_sampled_low,data_sampled_medium,data_sampled_high)

t_classify_sampled_inc<-proc.time()
model_classify_sampled_inc<-train(H_cat~.,data=select(data_sampled,-Id,-Hazard),
                              model="rf",tuneGrid = rfgrid,
                              trControl = trainControl(method = "oob"))
proc.time() - t_classify_sampled_inc

model_classify_sampled_inc$finalModel

save.image("092815.RData")


data_sampled_low<-lib_train%>%
  filter(H_cat=="low")%>%
  sample_n(250)

data_sampled_medium<-lib_train%>%
  filter(H_cat=="medium")%>%
  sample_n(250)
data_sampled_high<-lib_train%>%
  filter(H_cat=="high")%>%
  sample_n(250)
data_sampled<-rbind(data_sampled_low,data_sampled_medium,data_sampled_high)

model_rf_sampled<-train(Hazard~.,data=select(data_sampled,-Id,-H_cat), 
                        model="rf",tuneGrid = rfgrid,
                        trControl = trainControl(method = "oob"))
  
model_rf_sampled_inc<-train(Hazard~.,data=select(data_sampled_inc,-Id,-H_cat), 
                            model="rf",tuneGrid = rfgrid,
                            trControl = trainControl(method = "oob"))
#### GENERATE SMOTE SAMPLE


data_smote<-SMOTE(H_cat~.,data=lib_train,k=5,perc.over=10000,perc.under = 100)
data_smote<-SMOTE(H_cat~.,data=data_smote,k=5,perc.over=5000,perc.under = 200)
data_smote<-data_smote%>%
  group_by(H_cat) %>%
  sample_n(10000)
summary(data_smote$H_cat)

ggplot(aes(x=Hazard),data=data_smote)+
  geom_bar(binwidth=1)
temp<-select(ungroup(data_smote),-Id,-H_cat)
model_rf_smote<-train(Hazard~.,data=temp, 
                            model="rf",tuneGrid = rfgrid,
                            trControl = trainControl(method = "oob"))
model_rf_smote_test<-predict(model_rf_smote,newdata = lib_test_validation)
RMSE(model_rf_smote_test,lib_test_validation$Hazard)


#### GENERATE SMOTE2 SAMPLE


data_smote2<-SMOTE(H_cat~.,data=lib_train,k=5,perc.over=2000,perc.under = 100)
data_smote2<-SMOTE(H_cat~.,data=data_smote2,k=5,perc.over=1000,perc.under = 200)


ggplot(aes(x=Hazard),data=data_smote2)+
  geom_bar(binwidth=1)
temp<-select(data_smote2,-Id,-H_cat)
model_rf_smote2<-train(Hazard~.,data=temp, 
                      model="rf",tuneGrid = rfgrid,
                      trControl = trainControl(method = "oob"))
model_rf_smote2_test<-predict(model_rf_smote2,newdata = lib_test_validation)
RMSE(model_rf_smote2_test,lib_test_validation$Hazard)
#### HISTOGRAM PLOTS

p1<-ggplot(aes(x=Hazard),data=lib_train)+
  geom_bar()
p1


p1<-ggplot(aes(x=Hazard),data=lib_train)+
  geom_bar(binwidth = 1, colour = rgb(red=0,green=0,blue=0), 
           fill = rgb(red=0,green=1,blue=0))+
  ggtitle("no sampling high hazard")+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20),
        axis.title = element_text(lineheight=.8, face="bold",size=15),
        axis.text = element_text(lineheight=.8, face="bold",size=12))+
  ylim(0,100)+
  xlim(20,70)
p1

p2<-ggplot(aes(x=Hazard),data=data_sampled)+
  geom_bar(binwidth = 1, colour = rgb(red=0,green=0,blue=0), 
           fill = rgb(red=0,green=1,blue=0))+
  ggtitle("under sampling")+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20),
        axis.title = element_text(lineheight=.8, face="bold",size=15),
        axis.text = element_text(lineheight=.8, face="bold",size=12))
p2

p3<-ggplot(aes(x=Hazard),data=data_sampled_inc)+
  geom_bar(binwidth = 1, colour = rgb(red=0,green=0,blue=0), 
           fill = rgb(red=0,green=1,blue=0))+
  ggtitle("over sampling")+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20),
        axis.title = element_text(lineheight=.8, face="bold",size=15),
        axis.text = element_text(lineheight=.8, face="bold",size=12))
p3

p4<-ggplot(aes(x=Hazard),data=data_smote2)+
  geom_bar(binwidth = 1, colour = rgb(red=0,green=0,blue=0), 
           fill = rgb(red=0,green=1,blue=0))+
  ggtitle("SMOTE")+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20),
        axis.title = element_text(lineheight=.8, face="bold",size=15),
        axis.text = element_text(lineheight=.8, face="bold",size=12))
p4

library(gridExtra)

grid.arrange(p1,p2,p3,p4,nrow=2,
  main=textGrob("Hazard Distribution For Different Sampling Techniques",gp=gpar(fontsize=20,font=1,col="blue")))


tiff("histograms_blog.tiff",height=10 , width = 10 , units= 'in' , res=300)
grid.arrange(p1,p2,p3,p4,nrow=2,
             main=textGrob("Hazard Distribution For Different Sampling Techniques",gp=gpar(fontsize=40,font=1,col="blue")))
dev.off()

p5<-ggplot(aes(x=Hazard),data=lib_train)+
  geom_bar(binwidth = 1, colour = rgb(red=0,green=0,blue=0), 
           fill = rgb(red=0,green=1,blue=0))+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20),
        axis.title = element_text(lineheight=.8, face="bold",size=15),
        axis.text = element_text(lineheight=.8, face="bold",size=12))

p5



######RESULTS AND POST PROCESSING


temp_low<-filter(lib_test_validation,H_cat=="low")
temp_medium<-filter(lib_test_validation,H_cat=="medium")
temp_high<-filter(lib_test_validation,H_cat=="high")
> rmse_rf<-c(RMSE(temp_low$rf,temp_low$Hazard),
             +             RMSE(temp_medium$rf,temp_medium$Hazard),
             +             RMSE(temp_high$rf,temp_high$Hazard))
> rmse_rf_sampled<-c(RMSE(temp_low$rf_sampled,temp_low$Hazard),
                     +             RMSE(temp_medium$rf_sampled,temp_medium$Hazard),
                     +             RMSE(temp_high$rf_sampled,temp_high$Hazard))
> rmse_rf_sampled_inc<-c(RMSE(temp_low$rf_sampled_inc,temp_low$Hazard),
                         +             RMSE(temp_medium$rf_sampled_inc,temp_medium$Hazard),
                         +             RMSE(temp_high$rf_sampled_inc,temp_high$Hazard))
rmse_rf_smote2<-c(RMSE(temp_low$rf_smote2,temp_low$Hazard),
                             RMSE(temp_medium$rf_smote2,temp_medium$Hazard),
                            RMSE(temp_high$rf_smote2,temp_high$Hazard))
###HAZARD VS PREDICTED HAZARD PLOTS#####
################


p1 <- ggplot(aes(x=Hazard, y = pred), data = lib_train)+
  geom_point(alpha = .1)+
  ggtitle("no sampling")+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20, colour = "GREEN"),
        axis.title = element_text(lineheight=.8, face="bold",size=15, colour = "blue"),
        axis.text = element_text(lineheight=.8, face="bold",size=12))+
  ylim(0,70)+
  xlim(0,70)+
  xlab("Actual ")+
  ylab("Predicted ")
p1

p2 <- ggplot(aes(x=Hazard, y = pred), data = data_sampled)+
  geom_point(alpha = 1)+
  ggtitle("under sampling")+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20, colour = "GREEN"),
        axis.title = element_text(lineheight=.8, face="bold",size=15, colour = "blue"),
        axis.text = element_text(lineheight=.8, face="bold",size=12))+
  ylim(0,70)+
  xlim(0,70)+
  xlab("Actual ")+
  ylab("Predicted ")
p2
p3 <- ggplot(aes(x=Hazard, y = pred), data = data_sampled_inc)+
  geom_point(alpha = .1)+
  ggtitle("over sampling ")+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20,colour = "GREEN"),
        axis.title = element_text(lineheight=.8, face="bold",size=15,colour = "blue"),
        axis.text = element_text(lineheight=.8, face="bold",size=12))+
  ylim(0,70)+
  xlim(0,70)+
  xlab("Actual ")+
  ylab("Predicted ")
p3

p4 <- ggplot(aes(x=Hazard, y = pred), data = data_smote2)+
  geom_point(alpha = .1)+
  ggtitle("SMOTE")+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20, colour = "GREEN"),
        axis.title = element_text(lineheight=.8, face="bold",size=15 , colour = "blue"),
        axis.text = element_text(lineheight=.8, face="bold",size=12))+
  ylim(0,70)+
  xlim(0,70)+
  xlab("Actual ")+
  ylab("Predicted ")
p4


library(gridExtra)

grid.arrange(p1,p2,p3,p4,nrow=2,
             main=textGrob("Predicted vs. actual Hazards For Different Sampling Techniques",gp=gpar(fontsize=20,font=1,col="red")))


tiff("histograms_blog.tiff",height=10 , width = 10 , units= 'in' , res=300)
grid.arrange(p1,p2,p3,p4,nrow=2,
             main=textGrob("Predicted vs. actual Hazards For Different Sampling Techniques",
                           gp=gpar(fontsize=40,font=1,col="red")))
dev.off()

p5<-ggplot(aes(x=Hazard),data=lib_train)+
  geom_bar(binwidth = 1, colour = rgb(red=0,green=0,blue=0), 
           fill = rgb(red=0,green=1,blue=0))+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20),
        axis.title = element_text(lineheight=.8, face="bold",size=15),
        axis.text = element_text(lineheight=.8, face="bold",size=12))

p5

###ENSEMBLE#####
################
set.seed(1141)
data_sampled_low<-lib_train%>%
  filter(H_cat=="low")%>%
  sample_n(10000)

data_sampled_medium<-lib_train%>%
  filter(H_cat=="medium")%>%
  sample_n(10000,replace = TRUE)
data_sampled_high<-lib_train%>%
  filter(H_cat=="high")%>%
  sample_n(10000,replace=TRUE)
data_sampled_ensemble<-rbind(data_sampled_low,data_sampled_medium,data_sampled_high)
### classification###
model_classify_ensemble<-train(H_cat~.,data=filter(data_sampled_ensemble,-Id,-Hazard),
                              model="rf",tuneGrid = rfgrid,
                              trControl = trainControl(method = "oob"))
classification_data<-model_classify_ensemble$trainingData
classification_data<-mutate(classification_data,Hazard=data_sampled_ensemble$Hazard)
##Hazard evaluation###

model_rf_ensemble_low<-train(Hazard~.,data=select(filter(classification_data,.outcome=="low"),-.outcome),
                    model="rf",tuneGrid = rfgrid,
                    trControl = trainControl(method = "oob"))
print(model_rf_low)

model_rf_ensemble_medium<-train(Hazard~.,data=select(filter(classification_data,.outcome=="medium"),-.outcome),
                       model="rf",tuneGrid = rfgrid,
                       trControl = trainControl(method = "oob"))
print(model_rf_medium)


model_rf_ensemble_high<-train(Hazard~.,data=filter(classification_data,.outcome=="high"),-.outcome),
                     model="rf",tuneGrid = rfgrid,
                     trControl = trainControl(method = "oob"))
print(model_rf_high)




save.image("092915.RData")

overall_RMSE<-
model_rf_ensemble_low*sqrt(nrow(filter(classificataion_data,.outcome=="low")))+
model_rf_ensemble_medium*sqrt(nrow(filter(classificataion_data,.outcome=="medium")))+
model_rf_ensemble_high*sqrt(nrow(filter(classificataion_data,.outcome=="high")))

overall_RMSE<-overall_RMSE*sqrt((1/nrow(classification_data)))



p_testsetresults <- ggplot(aes(x=rowname,y=value,fill=variable),data=results)+
   geom_bar(stat="identity",position="dodge")+
   ggtitle("Test Set Results")+
   xlab("Hazard Category")+
   ylab("RMSE")+
   scale_fill_manual(values=c("darkred", "blue", "green" , "orange"), 
                    name="Model",
                   
                    labels=c("w/o sampling", "undersampling", "oversampling" , "SMOTE"))+
  theme(plot.title = element_text(lineheight=.8, face="bold", size=20),
  axis.title = element_text(lineheight=.8, face="bold",size=15),
  axis.text = element_text(lineheight=.8, face="bold",size=14))










######
library(GGally)
ggpairs(lib_train_sample)
t_ggp <- proc.time()
p_full<-  ggpairs(lib_train_1,
                  upper =list(continuous="points",combo="box",discrete="histogram"),
                  alpha=1/10)
proc.time() - t_ggp






### EDA
library(ggplot2)
p1<-ggplot(aes(x=Hazard),data=lib_train)+
  geom_bar(binwidth=2,fill="lightgreen", colour="black")
p1


p2<-ggplot(aes(y=Hazard,x=value),data=filter(lib_train_long,H_cat=="low"))+
  geom_point(alpha=1/5)+
  facet_wrap(~TV,scales="free")
p2


p3<-ggplot(aes(x=value),data=lib_train_long)+
  geom_bar(aes(fill=H_cat))+
  facet_wrap(~TV,scales="free")
p3

library(ggplot2)
tiff("HazardvsTV1.tiff",height=10 , width = 10 , units= 'in' , res=300)
p3<-ggplot(aes(x=value,y=Hazard),data=lib_train_long)+
  geom_boxplot(outlier.size=0.1)+
  scale_y_continuous(limits=c(0,70))+
  facet_wrap(~TV,ncol=6, scales = "free_x")
p3
dev.off()

p4<-ggplot(aes(x=value),data=filter(lib_train_long,Hazard>10))+
  +
  geom_bar(aes(fill=H_cat))+
  facet_wrap(~TV,scales="free")
p4


lib_train_long_low<-filter(lib_train_long,Hazard<11)
breaks_low=seq(1,10,1)
lib_train_long_low$H_cat_low<-cut(lib_train_long_low$Hazard,breaks_low,include.lowest = TRUE,
                                  right=FALSE)

p5<-ggplot(aes(x=value),data=lib_train_long_low)+
  geom_bar(aes(fill=H_cat_low))+
  facet_wrap(~TV,scales="free")
p5


##LESS VARIABLES MORE TREES
library(randomForest)

t_m_red <- proc.time()
m_red<-randomForest(Hazard~.,data=select(lib_train,-Id,-T1_V7,-T1_V8,-T1_V12,-T1_V15,-T2_V8,-H_cat),ntree=50,mtry=10)
proc.time() - t_m_red

lib_test_predicted_red<-predict(m_red,lib_test,type="response")
lib_test_predicted_red<-round(lib_test_predicted_red)
write.csv(lib_test_predicted_red,file="test_predict_rf_red_50_10.csv",append=FALSE)

library(caret)
t_caret_rf<-proc.time()
rf_model<-train(Hazard~.,data=select(lib_train,-Id,-H_cat),method="rf",
                trControl=trainControl(method="cv",number=5))
proc.time() - t_caret_rf

summary(rf_model)
lib_test_predicted_rf_cv<-predict(rf_model,lib_test,type="response")
lib_test_predicted_rf_cv<-round(lib_test_predicted_red)
write.csv(lib_test_predicted_rf_cv,file="test_predict_rf_model_cv.csv",append=FALSE)

t_caret_glm<-proc.time()
model_glm_bc_pca<-train(Hazard~.,data=select(lib_train,-Id,-H_cat),method="glm",preProcess="pca")
proc.time() - t_caret_glm


lib_test_predicted_glm<-predict(model_glm_bc,lib_test,type="raw")
lib_test_predicted_glm<-round(lib_test_predicted_glm)
write.csv(lib_test_predicted_glm,file="test_predict_glm.csv",append=FALSE)

init_model<-lm(Hazard~.,data=select(lib_train,-Id,-H_cat))
model_glm_refined<-stepAIC(init_model,direction="both")


t_caret_gam<-proc.time()
model_gam<-train(Hazard~.,data=select(lib_train,-Id,-H_cat),method="gam")
proc.time() - t_caret_gam


library(caret)
t_rf_cat<-proc.time()
model_rf_cat<-train(H_cat~.,data=select(data_sampled,-Id,-Hazard),method="rf")
proc.time() - t_rf_cat
model_rf_cat

t_rf_cat<-proc.time()
model_rf_cat_refined<-stepAIC(model_rf_cat)
proc.time() - t_rf_cat

p5<-ggplot(aes(x=value,y=Hazard),data=filter(select(filter(lib_train_long,H_cat=="high"),-H_cat)))+
  geom_point(alpha=1/5)+
  facet_wrap(~TV)
p5

t_glm_medium<-proc.time()
model_glm_medium<-lm(Hazard~T1_V1+T1_V3+T1_V4+T1_V8+T1_V15+T1_V16+T2_V9+T2_V13,data=filter(select(filter(lib_train,H_cat=="medium"),-H_cat,-Id)))
proc.time() - t_glm_medium

t_lm_high<-proc.time()
model_lm_high_ref<-stepAIC(model_lm_high,direction="both",steps=10000)
proc.time() - t_lm_high

model_lm_high<-train((Hazard)~.,data=filter(select(filter(lib_train,H_cat=="high"),-H_cat,-Id)),preProcess="expoTrans",method="rf")


## multi collinearity plot test
library(dplyr)
data_mp<-select(lib_train,-Id)
library(GGally)
ggpairs(data_mp)

tiff( "plot1.tiff", width = 600, height = 600 )
p<-pairs(data_mp,font.labels = 2, main="scatter plot matrix for liberty mutual data", 
         line.main=2,horInd = 1:2, verInd = 1:2,
         pch=21,col=rgb(red=1,blue=0,green=0,alpha=0.4),
         bg=rgb(red=1,blue=0,green=0,alpha=0.4))
dev.off()


tiff( "plot_cat_mres.tiff", height = 15, width = 15, units = 'in', res=100 )
p_cat<-pairs(data_mp,font.labels = 2, main="scatter plot matrix with category for liberty mutual data", 
             line.main=2,
             pch=21,col=c("blue","green3","red")[unclass(data_mp$H_cat)],
             bg=c("blue","green3","red")[unclass(data_mp$H_cat)])
dev.off()


tiff( "plot_alpha_p1.tiff", height = 30, width = 30, units = 'in', res=300 )
p_alpha<-pairs(data_mp,font.labels = 2, main="scatter plot matrix for liberty mutual data (alpha=0.1)", 
               line.main=2,
               pch=21,col=rgb(red=1,blue=0,green=0,alpha=0.1),
               bg=rgb(red=1,blue=0,green=0,alpha=0.1))
dev.off()


library(ggplot2)
library(GGally)
tiff( "plot_alpha_1.tiff", height = 30, width = 30, units = 'in', res=300 )
p_alpha<-ggpairs(data_mp, title="scatter plot matrix for liberty mutual data", 
                 diag=list(continuous="bar",discrete="bar",line.main=2,
                           pch=21,col=rgb(red=1,blue=0,green=0,alpha=0.4),
                           bg=rgb(red=1,blue=0,green=0,alpha=0.4))
                 dev.off()
                 
                 tiff( "plot_all.tiff", height = 30, width = 30, units = 'in', res=300)
                 p_all<-pairs(data_mp,main="scatter plot matrix for liberty mutual data",
                              font.labels = 2)
                 dev.off()
                 
                 library(dplyr)
                 data_sampled_low<-lib_train%>%
                   filter(H_cat=="low")%>%
                   sample_frac(.01)
                 
                 data_sampled_medium<-lib_train%>%
                   filter(H_cat=="medium")%>%
                   sample_frac(0.1)
                 
                 
                 data_sampled<-rbind(data_sampled_low,data_sampled_medium,filter(lib_train,Hazard>20))
                 
                 
                 tiff( "plot_alpha_sampled.tiff", height = 30, width = 30, units = 'in', res=300 )
                 p_alpha<-pairs(select(data_sampled,-Id),font.labels = 2, main="scatter plot matrix for sampled liberty mutual data (alpha=0.1)", 
                                line.main=2, lower.panel = NULL,
                                pch=21,col=NULL,
                                bg=rgb(red=1,blue=0,green=0,alpha=0.1))
                 dev.off()
                 
                 tiff( "plot_cat_sampled.tiff", height = 30, width = 30, units = 'in', res=300 )
                 p_alpha<-pairs(select(data_sampled,-Id),font.labels = 2, main="scatter plot matrix for sampled liberty mutual data with categories", 
                                line.main=2, lower.panel = NULL,
                                pch=21,col=NULL,
                                bg=c("blue","green3","red")[unclass(data_sampled$H_cat)])
                 dev.off()
                 
                 tiff( "plot_alpha_sampled.tiff", height = 30, width = 30, units = 'in', res=300 )
                 p_alpha<-ggpairs(select(data_sampled,-Id))
                 dev.off()
                 
                 tiff( "plot_cat_sampled.tiff", height = 30, width = 30, units = 'in', res=300 )
                 p_alpha<-pairs(select(data_sampled,-Id),font.labels = 2, main="scatter plot matrix for sampled liberty mutual data with categories", 
                                line.main=2, lower.panel = NULL,
                                pch=21,col=NULL,
                                bg=c("blue","green3","red")[unclass(data_sampled$H_cat)])
                 dev.off()
                 