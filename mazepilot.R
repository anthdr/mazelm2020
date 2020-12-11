rm(list=ls())
setwd("C:/Users/antoi/Google Drive/STAGE/R")

rawdata <- read.csv( "C:/Users/antoi/Google Drive/STAGE/R/resultspilot.txt" , sep=",", fill = TRUE , header = FALSE , comment.char="#", col.names = paste0("V",seq_len(15)))

names(rawdata) <- c("Time", "hash",  "Controller","Item",  "Element",  "Condition", "Group",  "Wordnumber",  "Word",  "Alternative",  "Word0=left,1=right",  "Correct",  "RT",  "Sentence",  "Total")

rawdata$RT[rawdata$RT == 'None'] <- NA
rawdata$RT <- as.numeric(rawdata$RT)

{
  rawdata$id <- NA
  timef <- as.factor(rawdata$Time)
  lvls <- levels(timef)
  
  for (i in 1:length(lvls)){
    
    rawdata[rawdata$Time==lvls[i] , ]$id <- i
    
  }
  
  rawdata$id <- as.factor(rawdata$id)
  nlevels(rawdata$id)
  rawdata$id <- as.numeric(rawdata$id)
}

rawdata <- subset(rawdata, id != 1 & id != 2)

#infos annexes
{
  participants <- subset(rawdata, rawdata$Condition == "intro")
  Mturk <- subset(rawdata, rawdata$Condition == "exit2")
  practice <- subset(rawdata, rawdata$Condition == "practice")
  filler <- subset(rawdata, rawdata$Condition == "filler")
}

data <- rawdata

data <- subset(data, !(Condition %in% "intro" | Condition %in% "practice" | Condition %in% "filler" | Condition %in% "exit2"))
data <- subset(data, data$Word == "has" | data$Word == "have")

filler.last.word <- filler
filler.last.word$end <- NA
filler.last.word$end[grep('\\.',filler.last.word$Word)] <- "last"
filler.last.word <- subset(filler.last.word, filler.last.word$end == 'last')

data$lastmatch <- NA
data$RC <- NA

data$lastmatch[grep("0",data$Condition)] <- "0"
data$lastmatch[grep("1",data$Condition)] <- "1"
data$RC[grep("1RC",data$Condition)] <- "1"
data$RC[grep("2RC",data$Condition)] <- "2"
data$RC[grep("3RC",data$Condition)] <- "3"

data.good <- subset(data, (Correct %in% 'yes'))

library(sciplot)
Condition <- c('1RC0lasmatch', '1RC1lasmatch', '2RC0lasmatch', '2RC1lasmatch', '3RC0lasmatch', '3RC1lasmatch') 

bargraph.CI(x.factor=RC,response=RT,group=lastmatch, col=c("tomato", "dark blue"), xlab="RC",ylab="RT",legend=TRUE,  x.leg=3,y.leg=1 , cex.leg=2, cex.names=2, cex.lab = 2, ylim = c(0,2000), data=data.good)
results <- data.frame(Condition)
results$RT.mean <- NA
results$RT.mean[results$Condition == '1RC0lasmatch'] <- mean(data.good$RT[data.good$RC == "1" & data.good$lastmatch == "0"])
results$RT.mean[results$Condition == '1RC1lasmatch'] <- mean(data.good$RT[data.good$RC == "1" & data.good$lastmatch == "1"])
results$RT.mean[results$Condition == '2RC0lasmatch'] <- mean(data.good$RT[data.good$RC == "2" & data.good$lastmatch == "0"])
results$RT.mean[results$Condition == '2RC1lasmatch'] <- mean(data.good$RT[data.good$RC == "2" & data.good$lastmatch == "1"])
results$RT.mean[results$Condition == '3RC0lasmatch'] <- mean(data.good$RT[data.good$RC == "3" & data.good$lastmatch == "0"])
results$RT.mean[results$Condition == '3RC1lasmatch'] <- mean(data.good$RT[data.good$RC == "3" & data.good$lastmatch == "1"])


data.good.has <- subset(data.good, (Word %in% 'has'))
bargraph.CI(x.factor=RC,response=RT,group=lastmatch, col=c("tomato", "dark blue"), xlab="RC, has",ylab="RT",legend=TRUE,  x.leg=3,y.leg=1 , cex.leg=2, cex.names=2, cex.lab = 2, ylim = c(0,2000), data=data.good.has)
results.has <- data.frame(Condition)
results.has$RT.mean[results.has$Condition == '1RC0lasmatch'] <- mean(data.good.has$RT[data.good.has$RC == "1" & data.good.has$lastmatch == "0"])
results.has$RT.mean[results.has$Condition == '1RC1lasmatch'] <- mean(data.good.has$RT[data.good.has$RC == "1" & data.good.has$lastmatch == "1"])
results.has$RT.mean[results.has$Condition == '2RC0lasmatch'] <- mean(data.good.has$RT[data.good.has$RC == "2" & data.good.has$lastmatch == "0"])
results.has$RT.mean[results.has$Condition == '2RC1lasmatch'] <- mean(data.good.has$RT[data.good.has$RC == "2" & data.good.has$lastmatch == "1"])
results.has$RT.mean[results.has$Condition == '3RC0lasmatch'] <- mean(data.good.has$RT[data.good.has$RC == "3" & data.good.has$lastmatch == "0"])
results.has$RT.mean[results.has$Condition == '3RC1lasmatch'] <- mean(data.good.has$RT[data.good.has$RC == "3" & data.good.has$lastmatch == "1"])

data.good.have <- subset(data.good, (Word %in% 'have'))
bargraph.CI(x.factor=RC,response=RT,group=lastmatch, col=c("tomato", "dark blue"), xlab="RC, have",ylab="RT",legend=TRUE,  x.leg=3,y.leg=1 , cex.leg=2, cex.names=2, cex.lab = 2, ylim = c(0,2000), data=data.good.have)
results.have <- data.frame(Condition)
results.have$RT.mean[results.have$Condition == '1RC0lasmatch'] <- mean(data.good.have$RT[data.good.have$RC == "1" & data.good.have$lastmatch == "0"])
results.have$RT.mean[results.have$Condition == '1RC1lasmatch'] <- mean(data.good.have$RT[data.good.have$RC == "1" & data.good.have$lastmatch == "1"])
results.have$RT.mean[results.have$Condition == '2RC0lasmatch'] <- mean(data.good.have$RT[data.good.have$RC == "2" & data.good.have$lastmatch == "0"])
results.have$RT.mean[results.have$Condition == '2RC1lasmatch'] <- mean(data.good.have$RT[data.good.have$RC == "2" & data.good.have$lastmatch == "1"])
results.have$RT.mean[results.have$Condition == '3RC0lasmatch'] <- mean(data.good.have$RT[data.good.have$RC == "3" & data.good.have$lastmatch == "0"])
results.have$RT.mean[results.have$Condition == '3RC1lasmatch'] <- mean(data.good.have$RT[data.good.have$RC == "3" & data.good.have$lastmatch == "1"])

