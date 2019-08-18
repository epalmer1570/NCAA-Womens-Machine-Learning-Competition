##############################################################################################
# Created for Google Machine learning competition
# Final place 217/500 participants
# My graded Log loss score of .40367, winner scored .32245 log loss


#Background: On a scale of 0 to 1 predict percentage chance of a team winning a given tournament matchup for all potential matchups

#Data: sets utilized will utilize regular season womens game by game stat numbers to predict performance in tournament
#Train Data: 2010-2018 womens regular season data as predicting (independent) variables -- use tournament performance as dependent variable
#Test Data: 2019 regular season data - predict winning percentage chance by all 2016 potential games (64 teams *63 potential opponents/ 2 teams per game)
##############################################################################################



###
# load packages
###
library(readxl)
library(plyr)
library(dplyr)
library(data.table)
library(randomForest)
library(ggplot2)




#Clear workspace
rm(list = ls())
#Formatting choice to show numbers not in scientific
options(scipen = 999)




###
# Read in files
###

Seeds <- read.csv("WNCAATourneySeeds.csv")
RegularSeason <- read.csv("WRegularSeasonDetailedResults.csv")


#*********************************************************************************************
#Section 1 - data manipulation and feature engineering based on a single teams standalone statistics
#*********************************************************************************************


##############################################################################################
# Section 1a. : Strength of schedule data & data manipulation
# Purpose of this section is to create a strength of schedule measure 
# Will be broken out by upset poential - i.e. how many quality ranked wins a team has
# Also will take into account consistency - i.e. general quality of losses, are they against good or bad teams
###############################################################################################


###
# Data is originally not split up by winners and losers rather home and away teams
# Lets make our data so instead of winners/losers we look at team vs opposing team with a given result (W/L). Our row count will need to double but this will give us more flexibilitys
###
columns <- colnames(RegularSeason)
beginning <- substr(columns,1,1)
  
WinningTeamColumn <- ifelse(beginning == "W","Team.",ifelse(beginning == "L","Opp.",beginning))
LosingTeamColumn <- ifelse(beginning == "L","Team.",ifelse(beginning == "W","Opp.",beginning))

columnRest <- substring(columns,2)
WinningTeamColumn <- paste(WinningTeamColumn,columnRest,sep = "")
LosingTeamColumn <- paste(LosingTeamColumn,columnRest,sep = "")

colnames(RegularSeason) <- WinningTeamColumn
WinningTeamdf <- RegularSeason
colnames(RegularSeason) <- LosingTeamColumn
LosingTeamdf <- RegularSeason

WinningTeamdf$Result <- "W"
LosingTeamdf$Result <- "L"

WinningTeamdf$Wins <- 1
LosingTeamdf$Losses <-1

#Join with Seeds to see total seeded wins

Seeds$Seed <- as.numeric(substr(Seeds$Seed,2,3))
Seeds$Key <- paste(Seeds$Season,Seeds$TeamID,sep = "-")
WinningTeamdf$Key <- paste(WinningTeamdf$Season,WinningTeamdf$Opp.TeamID,sep = "-")
LosingTeamdf$Key <- paste(LosingTeamdf$Season,LosingTeamdf$Opp.TeamID,sep = "-")
WinningTeamdf <- merge(WinningTeamdf,Seeds,by = "Key",all.x = T )
LosingTeamdf <- merge(LosingTeamdf,Seeds,by = "Key",all.x = T)

#Create Upset potential Consistency potential features
LosingTeamdf$Consistency.Potential <- ifelse(is.na(LosingTeamdf$Seed),32,LosingTeamdf$Seed)


test <- (16:1)
test2 <- (1:16)
test2 <- as.data.frame(test2)
test2$test <- test

WinningTeamdf <- merge(WinningTeamdf,test2,by.x = "Seed",by.y = "test2",all.x = T)
setnames(WinningTeamdf,"test","Upset.Potential")
WinningTeamdf$Upset.Potential <- ifelse(is.na(WinningTeamdf$Upset.Potential),0,WinningTeamdf$Upset.Potential)

Combined <- rbind.fill(WinningTeamdf,LosingTeamdf)


Combined$Seed <- NULL
Combined$Key <- paste(Combined$Season.x,Combined$Team.TeamID,sep = "-")
Combined <-merge(Combined,Seeds,by = "Key",all.x = T)
Combined[is.na(Combined)] <- 0

Combined$Season.y <- NULL
Combined <- Combined[, !(colnames(Combined) %in% c("TeamID.x","TeamID.y")), drop = FALSE]


###
#Group up Winning and Losing results
# Can combine data for raw stat numbers
# Can get data by games for wins vs losses
###

CombinedTeamGrouped <- Combined %>%
  group_by(Team.TeamID,Season.x,Seed) %>%
  summarise_if(is.numeric,sum,na.rm = T)

#exclude team stats for teams that missed tourney
CombinedTeamGrouped <- CombinedTeamGrouped[CombinedTeamGrouped$Seed!= 0,]

# Create Strength of Schedule Feature

CombinedTeamGrouped$Str.Of.Sched <- CombinedTeamGrouped$Upset.Potential/CombinedTeamGrouped$Consistency.Potential

# If strenth of schedule is infinite (undefeated team) we need to give it a number so we will make it the largest number in the data set
max <- max(CombinedTeamGrouped$Str.Of.Sched[is.finite(CombinedTeamGrouped$Str.Of.Sched)])
CombinedTeamGrouped$Str.Of.Sched <- ifelse(is.infinite(CombinedTeamGrouped$Str.Of.Sched),max,
                                           CombinedTeamGrouped$Str.Of.Sched)




##############################################################################################
# Section 1b. : Create features based on advanced basketball statistics
# Look at things like Field goal pecentage.
# In prior year challenge I simply used these features to create a simulated game score based on more advanced stats
##############################################################################################





################Compute Metrics
CombinedTeamGrouped$FG.Perc <- CombinedTeamGrouped$Team.FGM/CombinedTeamGrouped$Team.FGA
CombinedTeamGrouped$threept.Perc <- CombinedTeamGrouped$Team.FGM3/CombinedTeamGrouped$Team.FGA3
CombinedTeamGrouped$twopt.Perc <- (CombinedTeamGrouped$Team.FGM-CombinedTeamGrouped$Team.FGM3)/(CombinedTeamGrouped$Team.FGA-CombinedTeamGrouped$Team.FGA3)
CombinedTeamGrouped$opp3pt.Perc <- CombinedTeamGrouped$Opp.FGM3/CombinedTeamGrouped$Opp.FGA3
CombinedTeamGrouped$opp2pt.Perc <- (CombinedTeamGrouped$Opp.FGM-CombinedTeamGrouped$Opp.FGM3)/(CombinedTeamGrouped$Opp.FGA-CombinedTeamGrouped$Opp.FGA3)
CombinedTeamGrouped$FT.Perc <- CombinedTeamGrouped$Team.FTM/CombinedTeamGrouped$Team.FTA
CombinedTeamGrouped$oppFT.Perc <- CombinedTeamGrouped$Opp.FTM/CombinedTeamGrouped$Opp.FTA
##Weighted attempts of shots
CombinedTeamGrouped$w1 <- (CombinedTeamGrouped$Team.FGA-CombinedTeamGrouped$Team.FGA3)/CombinedTeamGrouped$Team.FGA
CombinedTeamGrouped$w2 <- CombinedTeamGrouped$Team.FGA3/CombinedTeamGrouped$Team.FGA
CombinedTeamGrouped$wFT <- CombinedTeamGrouped$Team.FTA/(CombinedTeamGrouped$Team.FGA*2)



CombinedTeamGrouped$AvgPossessions <-((CombinedTeamGrouped$Team.FGA-CombinedTeamGrouped$Opp.FGA)/(CombinedTeamGrouped$Team.FGA))+1



##Additional advanced stats
# Effective FG percentage and FGA differential, commonly used metrics by basketball analysts
CombinedTeamGrouped$FGA.Differential <- CombinedTeamGrouped$Team.FGA-CombinedTeamGrouped$Opp.FGA
CombinedTeamGrouped$Team.Effective.FG.Percent <- ((CombinedTeamGrouped$Team.FGM-CombinedTeamGrouped$Team.FGM3)+(1.5*CombinedTeamGrouped$Team.FGM3))/CombinedTeamGrouped$Team.FGA
CombinedTeamGrouped$Opp.Effective.FG.Percent  <- ((CombinedTeamGrouped$Opp.FGM-CombinedTeamGrouped$Opp.FGM3)+(1.5*CombinedTeamGrouped$Opp.FGM3))/CombinedTeamGrouped$Opp.FGA


#*********************************************************************************************
#Section 2 - Preparing data set for predictive model and feature engineering based on head to head statistics
#*********************************************************************************************


##############################################################################################
# Section 2a. : Create data matrix that computes all possible games
# Compute theoretical game scores (The last feature we will engineer for our data set (Section 1))
# Need to create all possible games by doing a join between the same data frames (Section 2)
##############################################################################################
rm(Combined,LosingTeamdf,test2,WinningTeamdf)
DataMatrix <- merge(CombinedTeamGrouped,CombinedTeamGrouped,by = NULL)


DataMatrix$TeamXOffense <- 100*DataMatrix$AvgPossessions.x*(2*DataMatrix$w1.x*((DataMatrix$twopt.Perc.x+DataMatrix$opp2pt.Perc.y)/2)
                                                            +3*DataMatrix$w2.x*((DataMatrix$threept.Perc.x+DataMatrix$opp3pt.Perc.y)/2)
                                                            +DataMatrix$wFT.x*DataMatrix$FT.Perc.x)

DataMatrix$TeamYOffense <- 100*DataMatrix$AvgPossessions.y*(2*DataMatrix$w1.y*((DataMatrix$twopt.Perc.y+DataMatrix$opp2pt.Perc.x)/2)
                                                            +3*DataMatrix$w2.y*((DataMatrix$threept.Perc.y+DataMatrix$opp3pt.Perc.x)/2)
                                                            +DataMatrix$wFT.y*DataMatrix$FT.Perc.y)


DataMatrix$OutcomeRaw <- DataMatrix$TeamXOffense-DataMatrix$TeamYOffense
DataMatrix <- DataMatrix[DataMatrix$Season.x.x == DataMatrix$Season.x.y,]



###
# Get regular season results - head to head results
# Can be used as a predictor in the tournament if there were previous head to head games
###

DataMatrix$Key2 <- paste(DataMatrix$Season.x.x,DataMatrix$Team.TeamID.x,DataMatrix$Team.TeamID.y,sep = "-")

RegularSeason <- read.csv("WRegularSeasonDetailedResults.csv")
RegularSeason$Key2 <- paste(RegularSeason$Season,RegularSeason$WTeamID,RegularSeason$LTeamID,sep = "-")

DataMatrix.Test <- merge(DataMatrix,RegularSeason,by = "Key2",all.x = T)


DataMatrix.Test <- DataMatrix.Test %>%
  group_by(WTeamID,LTeamID,Season) %>%
  summarise(Wins =n())

DataMatrix.Test$Team.TeamID.x <-ifelse(DataMatrix.Test$WTeamID <= DataMatrix.Test$LTeamID, DataMatrix.Test$WTeamID,DataMatrix.Test$LTeamID)
DataMatrix.Test$Team.TeamID.y <- ifelse(DataMatrix.Test$Team.TeamID.x == DataMatrix.Test$WTeamID,DataMatrix.Test$LTeamID,DataMatrix.Test$WTeamID)
DataMatrix.Test$Win.Loss <- ifelse(DataMatrix.Test$Team.TeamID.x == DataMatrix.Test$WTeamID,DataMatrix.Test$Wins,DataMatrix.Test$Wins*-1)

DataMatrix.Test <- DataMatrix.Test[,c("Season","Team.TeamID.x","Team.TeamID.y","Win.Loss")]
DataMatrix.Test$Key2 <- paste(DataMatrix.Test$Season,DataMatrix.Test$Team.TeamID.x,DataMatrix.Test$Team.TeamID.y,sep = "-")


DataMatrix.Test <- DataMatrix.Test %>%
  group_by(Key2) %>%
  summarise(Wins.Loss = sum(Win.Loss))



DataMatrix2 <- merge(DataMatrix,DataMatrix.Test,by = "Key2",all.x = T)
###
#Take out duplicate games, data matrix creates 2 identical games so just keep the rows with lower team id#
###
DataMatrix2 <- DataMatrix2[DataMatrix2$Team.TeamID.x < DataMatrix2$Team.TeamID.y,]


###
#Additional advanced metrics
# These are features that will compare team vs team stats
###
DataMatrix2$UpsetX.vs.ConsistencyY <- DataMatrix2$Upset.Potential.x+DataMatrix2$Consistency.Potential.y
DataMatrix2$ConsistencyX.vs.UpsetY <- DataMatrix2$Upset.Potential.y+DataMatrix2$Consistency.Potential.x

#Part of V2 adds
DataMatrix2$Seed.Differential <- DataMatrix2$Seed.x-DataMatrix2$Seed.y
DataMatrix2$Str.Of.Sched.Differential <- DataMatrix2$Str.Of.Sched.x-DataMatrix2$Str.Of.Sched.y
DataMatrix2$Upset.Potential.Differential <- DataMatrix2$Upset.Potential.x-DataMatrix2$Upset.Potential.y
DataMatrix2$Consistency.Potential.Differential <- DataMatrix2$Consistency.Potential.x-DataMatrix2$Consistency.Potential.y
DataMatrix2$Team.Effective.FG.Percent.Differential <- DataMatrix2$Team.Effective.FG.Percent.x-DataMatrix2$Team.Effective.FG.Percent.y
DataMatrix2$Opp.Effective.FG.Percent.Differential <- DataMatrix2$Opp.Effective.FG.Percent.x-DataMatrix2$Opp.Effective.FG.Percent.y
DataMatrix2$FGA.Differential.Game <- DataMatrix2$FGA.Differential.x-DataMatrix2$FGA.Differential.y
DataMatrix2$FG.Perc.Differential <- DataMatrix2$FG.Perc.x-DataMatrix2$FG.Perc.y
DataMatrix2$Team.Score.Differential <- DataMatrix2$Team.Score.x-DataMatrix2$Team.Score.y
DataMatrix2$Opp.Score.Differential <- DataMatrix2$Opp.Score.x-DataMatrix2$Opp.Score.y
DataMatrix2$Team.Ast.Differential <- DataMatrix2$Team.Ast.x-DataMatrix2$Team.Ast.y
DataMatrix2$Team.Blk.Differential <- DataMatrix2$Team.Blk.x-DataMatrix2$Team.Blk.y



#*********************************************************************************************
#Section 3 - Feature normalization and splitting datasets
#Note: some feature engineering is included in this section as well as a way to 
#*********************************************************************************************


##############################################################################################



###
# Read in tournament results for train
# These will ultimately be used to validate model
###
ResultsTrain <- read.csv("WNCAATourneyCompactResults.csv")
ResultsTrain$Win <- 1

ResultsTrain$Team.TeamID.x <-ifelse(ResultsTrain$WTeamID <= ResultsTrain$LTeamID, ResultsTrain$WTeamID,ResultsTrain$LTeamID)
ResultsTrain$Team.TeamID.y <- ifelse(ResultsTrain$Team.TeamID.x == ResultsTrain$WTeamID,ResultsTrain$LTeamID,ResultsTrain$WTeamID)
ResultsTrain$Result <- ifelse(ResultsTrain$Team.TeamID.x == ResultsTrain$WTeamID,ResultsTrain$Win,0)
ResultsTrain$Key2 <- paste(ResultsTrain$Season,ResultsTrain$Team.TeamID.x,ResultsTrain$Team.TeamID.y,sep = "-")
ResultsTrain <- ResultsTrain[ResultsTrain$Season >=2010,]

ResultsTrain <- ResultsTrain %>%
  group_by(Key2) %>%
  summarise(Result = sum(Result))

#JoinwithDataMatrix

DataMatrixFinal <- merge(ResultsTrain,DataMatrix2,by ="Key2",all.x = T)

##Add in this to get 2019 data
DataMatrix2019 <- DataMatrix2[DataMatrix2$Season.x.x ==2019,]
DataMatrix2019$Result <- 0

DataMatrixFinal <-rbind(DataMatrixFinal,DataMatrix2019)

###
# Normalize Features for supervised learning
###

#set NAs = 0
DataMatrixFinal[is.na(DataMatrixFinal)] <- 0

#Normalize
normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

DataMatrix.Normalized <- as.data.frame(lapply(DataMatrixFinal[2:ncol(DataMatrixFinal)],normalize))

###
# See if adding/removing features will  help my model
### 
DataMatrix.Normalized$Wins.Loss2 <- DataMatrixFinal$Wins.Loss
DataMatrix.Normalized$Wins.Loss.Binary <- ifelse(DataMatrix.Normalized$Wins.Loss2>0,1,ifelse(DataMatrix.Normalized$Wins.Loss2 <0,-1,0))
  #added 2 below since .440 log loss
DataMatrix.Normalized$PastWin <- ifelse(DataMatrix.Normalized$Wins.Loss2>0,1,0)
DataMatrix.Normalized$PastLoss <- ifelse(DataMatrix.Normalized$Wins.Loss2 <0,1,0)

DataMatrix.Normalized$Wins.Loss2 <- NULL
DataMatrix.Normalized$Consistency.Upset.MultiDirectional <- (DataMatrix.Normalized$UpsetX.vs.ConsistencyY)-(DataMatrix.Normalized$ConsistencyX.vs.UpsetY)
DataMatrix$Weighted.Outcome <- DataMatrix$OutcomeRaw+Str.Of.Sched.Differential

DataMatrix.Normalized <- as.data.frame(lapply(DataMatrix.Normalized[1:ncol(DataMatrix.Normalized)],normalize))
DataMatrix.Normalized$Season.x.x <- DataMatrixFinal$Season.x.x

DataMatrix.Normalized$Consistency.Upset.MultiDirectional.GuaranteedWin <- 
  ifelse(DataMatrix.Normalized$Consistency.Upset.MultiDirectional>.625,1,0)

DataMatrix.Normalized$Consistency.Upset.MultiDirectional.GuaranteedLoss <- 
  ifelse(DataMatrix.Normalized$Consistency.Upset.MultiDirectional<.375,1,0)
#Remove non predicting Features
colnames(DataMatrix.Normalized)


DataMatrix.Normalized <- DataMatrix.Normalized.Orig
DataMatrix.Normalized.Orig <- DataMatrix.Normalized
DataMatrix.Normalized <- DataMatrix.Normalized.Orig
    #added below since .440 log loss
EP<- DataMatrix.Normalized[,  c("Season.y","Season.x.y","Season.x","Opp.TeamID.x","Opp.TeamID.y","Team.TeamID.x","Team.TeamID.y",
                                "DayNum.x","DayNum.y")]
DataMatrix.Normalized <- DataMatrix.Normalized[, !(colnames(DataMatrix.Normalized) %in% c("Season.y","Season.x.y","Season.x","Opp.TeamID.x","Opp.TeamID.y",
                                                                                          "DayNum.x","DayNum.y")), drop = FALSE]

DataMatrix.Normalized$Team.TeamID.x <- DataMatrixFinal$Team.TeamID.x
DataMatrix.Normalized$Team.TeamID.y <- DataMatrixFinal$Team.TeamID.y
DataMatrix.Normalized$Key2 <- DataMatrixFinal$Key2

###
# Run Random Forest
###


#
#Final Split of dataset
#Train = 2010-2017
#Test = 2018 data (see how model would have done in 2018)
#Test.2019 = 2019 data (i.e. my submission for the competition) 
#Note: For actual competition for the final Train data set i incorporated my 2018 data as well in my final algorithm

Train <- DataMatrix.Normalized[DataMatrix.Normalized$Season.x.x != 2019 & DataMatrix.Normalized$Season.x.x != 2018,]
Test <- DataMatrix.Normalized[DataMatrix.Normalized$Season.x.x ==2018,]
Test.2019 <- DataMatrix.Normalized[DataMatrix.Normalized$Season.x.x ==2019,]
DataMatrix.Normalized$Season.x.x <- NULL


# Set seed and run Random Forest model on train data
set.seed(1000)

randomforest <- randomForest(Result ~. -Team.TeamID.x-Team.TeamID.y-Key2,data = Train,ntree=2000,mtry =10,importance=TRUE) #reducing number of trees to 800 from 1000
randomforest
plot(randomforest)
  

# Run training model on test data


pred<-predict(randomforest,Test) #Predictions on Test Set for each Tree
varImpPlot(randomforest,type=2)
Test$pred <- pred

Test$difference.result <- abs(Test$pred - Test$Result)
Test$Log.Loss <- Test$Result*log(Test$pred)+(1-Test$Result)*log(1-Test$pred)
-mean(Test$Log.Loss) #.4534892 log loss in prior year this would have been good enough for top 55 out of 800 submissions



#Create 2019 File

pred<-predict(randomforest,Test.2019) #Predictions on Test Set for each Tree
varImpPlot(randomforest,type=2)
Test.2019$pred <- pred

Final <- Test.2019[,c("Key2","pred")]
Final <- Final[order(Final$Key2),]
Final$Key2 <- gsub("-","_",Final$Key2)
setnames(Final,"Key2","ID")
setnames(Final,"pred","Pred")



#  write.csv(Final,"Womens.ML.Output1_ALL FEATURES.csv")


#
ggplot(Test, aes(x = Seed.x, y = pred)) +
  geom_col()

qplot(Seed.Differential, pred, colour = (Seed.x), 
      data = Test)









###################################################
#Visuals used to analyze features used in new algorithm, or create new features
###################################################


#Scatterplot results Random Forest
Test$pred.Round <- round(Test$pred,0)
Test$Correct <- ifelse(Test$pred.Round-Test$Result == 0,1,0)
Test$n <- 1

qplot(pred,Consistency.Upset.MultiDirectional, colour = as.character(Correct), shape = as.character(Result), 
      data = Test)


#Histogram
#Analyze distribution of my submission predictions - We can see that most of my predictions are in the .9-.1 certainty range
ggplot(Test.2019, aes(x=pred)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white",bins = 10)



#Histogram of prediction confidence vs results - shows most high confidence picks yield better results
Test$probcut <- cut(Test$pred, seq(0, 1, 0.1))

Test.Histogram <- Test%>%
  group_by(probcut) %>%
  summarise(hprob_counts = n(), Correct = sum(Correct))

Test.Histogram$Correct.Rate <- Test.Histogram$Correct/Test.Histogram$hprob_counts

ggplot(Test.Histogram, aes(x = probcut, y = Correct.Rate)) +
  geom_col()

#Total correct
sum(Test.Histogram$Correct)/sum(Test.Histogram$hprob_counts)


###
# Results for Train
###

pred<-predict(randomforest,Train) #Predictions on Test Set for each Tree

Train$pred <- pred

Train$difference.result <- abs(Train$pred - Train$Result)
Train$Log.Loss <- Train$Result*log(Train$pred)+(1-Train$Result)*log(1-Train$pred)
-mean(Train$Log.Loss)



Train$pred.Round <- round(pred,0)
Train$Correct <- ifelse(Train$pred.Round-Train$Result == 0,1,0)
Train$n <- 1


#Scatterplot results Random Forest
qplot(Str.Of.Sched.Differential, pred, colour = as.character(Result), shape = as.character(Result), 
      data = Train)

#Scatterplot results Random Forest
##These visuals go together
# Proves out that a team with not a lot of ranked wins (upset potential) cannot beat a team that has consistently beaten worse competition (consistency)
#Consistency and upset and how it aligns with predictions and actual wis and losses
qplot(UpsetX.vs.ConsistencyY, pred, colour = as.character(Result), shape = as.character(Result), 
      data = Train)  # Great visual every team except  with a value over .5 has won

qplot(ConsistencyX.vs.UpsetY, pred, colour = as.character(Result), shape = as.character(Result), 
      data = Train)  # Great visual every team except  with a value over .5 has won


qplot((UpsetX.vs.ConsistencyY),(ConsistencyX.vs.UpsetY), colour = as.character(Result), shape = as.character(Result), 
      data = Train)  # Combining the visuals together to show a very low score always is loss and very high score always is a win



qplot(Consistency.Upset.MultiDirectional, pred, colour = as.character(Result), shape = as.character(Result), 
      data = Train)  # Great visual every team except  with a value over .5 has won


ggplot(Train, aes(x=Consistency.Upset.MultiDirectional)) + 
  geom_histogram(colour="black", fill="green",bins = 8)


qplot(log(UpsetX.vs.ConsistencyY),log(ConsistencyX.vs.UpsetY), colour = as.character(Result), shape = as.character(Result), 
      data = Train)  # Combining the visuals together


#Scatterplot Str. Of Schedule
qplot(Str.Of.Sched.Differential, Seed.Differential, colour = as.character(Result), shape = as.character(Result), 
      data = Train) 

qplot(Team.Score.Differential, Upset.Potential.Differential, colour = as.character(Result), shape = as.character(Result), 
      data = Train) 

qplot(OutcomeRaw, Str.Of.Sched.Differential, colour = as.character(Result), shape = as.character(Result), 
      data = Train) ###Make this a feature, top right quadrant mans win every time almost


#Head to head scatterplots
Train.HeadtoHead <- Train[Train$Wins.Loss != .50,]

qplot( pred, Wins.Loss,colour = as.character(Result), shape = as.character(Result), 
      data = Train.HeadtoHead) ###Make this a feature, top right quadrant mans win every time almost

Train.HeadtoHead$test <- 1
Train.HeadtoHead$HeadtoHeadvsTourneyresult <- ifelse(Train.HeadtoHead$Result == 1 & Train.HeadtoHead$Wins.Loss.Binary == 1, "Same Result",
                           ifelse(Train.HeadtoHead$Result == 0 & Train.HeadtoHead$Wins.Loss.Binary == 0 ,"Same Result","Different Result"))
Train.HeadtoHead$Result <- as.character(Train.HeadtoHead$Result)



ggplot(Train.HeadtoHead, aes( x= test,y = test, fill = HeadtoHeadvsTourneyresult)) + 
  geom_bar(position = "fill",stat = "identity") +
 
  scale_y_continuous(labels = scales::percent_format()) #viz shows that 75% of the time a team originally wins they will win again

