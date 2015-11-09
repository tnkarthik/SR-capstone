setwd("C:/Users/Karthik/Documents/GitHub/libmutual/")
lib_train<-read.csv("train.csv")
lib_test<-read.csv("test.csv")

library(dplyr)
library(randomForest)

lib_train_1<-select(lib_train,-Id)

t_m3 <- proc.time()
  m_3<-randomForest(Hazard~.,data=lib_train_1,ntree=50,mtry=10)
proc.time() - t_m3

lib_test_predicted_3<-predict(m_3,lib_test,type="response")
lib_test_predicted_3<-round(lib_test_predicted_3)
write.csv(lib_test_predicted_3,file="test_predict_rf_50_10.csv",append=FALSE)

lib_train_gpd<-group_by(lib_train,H_cat)
testcheck<-summarise(lib_train,n=n())
lib_train_sample<-sample_frac(lib_train_gpd,.01)
testcheck<-summarise(lib_train_sample,n=n())
lib_train_sample<-sample_frac(lib_train_1,.2)
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

breaks=c(0,10,20,70)

lib_train$H_cat<-cut(lib_train$Hazard,breaks,labels=c("low","medium","high"))

library(reshape2)

lib_train_long<-melt(lib_train_1,id.vars = "Hazard",variable.name = "TV")
###lib_train_long$value<-as.numeric(lib_train_long$value)

p2<-ggplot(aes(y=Hazard,x=value),data=filter(lib_train_long,H_cat=="low"))+
  geom_point(alpha=1/5)+
  facet_wrap(~TV,scales="free")
p2

lib_train_long$H_cat<-cut(lib_train_long$Hazard,breaks,labels=c("low","medium","high"))

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


tiff( "plot_cat.tiff", height = 30, width = 30, units = 'in', res=300 )
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
