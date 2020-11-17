# loading packages
library(tidyverse)
library(ParamHelpers)
library(mlr)
library(randomForest)
library(gbm)
library(GGally)
library(Hmisc)
library(corrplot)

# importing data
setwd(dirname(rstudioapi::getSourceEditorContext()$path)) #only works in Rstudio
semWI_credit <- read.csv("semWI_credit.csv", sep = ",")
semWI_credit$DEFAULT <- as.factor(semWI_credit$DEFAULT)

#-----------------------------------------------------------------------#
############################ Correlations ###############################
#-----------------------------------------------------------------------#
#Figure 4.1
corrplot(rcorr(as.matrix(semWI_credit))$r, type = "upper", tl.col = "black", tl.srt = 45) #if this doesnt work, make plot window bigger

#NumberRealEstateLoansOrLines <> DebtRatio 0.516728281
#NumberOfOpenCreditLinesAndLoands <> NumberRealEstate 0.421688817                  
#NumberOfOpenCreditLinesAndLoands <> DebtRatio 0.348378241



#-----------------------------------------------------------------------#
################################ MLR ####################################
#-----------------------------------------------------------------------#

# Removing PROFIT variable
semWI_credit = subset(semWI_credit, select = -c(PROFIT))

# creating task
creditTask <- makeClassifTask(data = semWI_credit, target = "DEFAULT", positive = "1")

# creating learners
learnerLogit <- makeLearner("classif.logreg", predict.type = "prob")
learnerRForest <- makeLearner("classif.randomForest", predict.type = "prob")
learnerGBM <- makeLearner("classif.gbm", predict.type = "prob")

# thresholding ----

# cost matrices & cost measures
costMatrix1 <- matrix(c(0, 2, 1, 0), 2)
colnames(costMatrix1) <- rownames(costMatrix1) <- getTaskClassLevels(creditTask)
creditCosts1<- makeCostMeasure(id = "creditCosts1", name = "Credit costs1", costs = costMatrix1, best = 0, worst = 2)

costMatrix2 <- matrix(c(0, 5, 1, 0), 2)
colnames(costMatrix2) <- rownames(costMatrix2) <- getTaskClassLevels(creditTask)
creditCosts2<- makeCostMeasure(id = "creditCosts2", name = "Credit costs2", costs = costMatrix2, best = 0, worst = 5)

costMatrix3 <- matrix(c(0, 10, 1, 0), 2)
colnames(costMatrix3) <- rownames(costMatrix3) <- getTaskClassLevels(creditTask)
creditCosts3<- makeCostMeasure(id = "creditCosts3", name = "Credit costs3", costs = costMatrix3, best = 0, worst = 10)

costMatrix4 <- matrix(c(0, 20, 1, 0), 2)
colnames(costMatrix4) <- rownames(costMatrix4) <- getTaskClassLevels(creditTask)
creditCosts4<- makeCostMeasure(id = "creditCosts4", name = "Credit costs4", costs = costMatrix4, best = 0, worst = 20)

costMatrix5 <- matrix(c(0, 50, 1, 0), 2)
colnames(costMatrix5) <- rownames(costMatrix5) <- getTaskClassLevels(creditTask)
creditCosts5<- makeCostMeasure(id = "creditCosts5", name = "Credit costs5", costs = costMatrix5, best = 0, worst = 50)


# Cross Validation
set.seed(123) #to have same results as in paper
rin = makeResampleInstance("CV", iters = 10, task = creditTask)
rLogit = resample(learnerLogit, creditTask, resampling = rin, measures = list(acc, fnr), show.info = FALSE)
rRForest = resample(learnerRForest, creditTask, resampling = rin, measures = list(acc, fnr), show.info = FALSE)
rGBM = resample(learnerGBM, creditTask, resampling = rin, measures = list(acc, fnr), show.info = FALSE)


# benchmarking the different learners by the new cost measure (with threshold = 0.5)
perfLogit <- performance(rLogit$pred, measures = list(creditCosts1, creditCosts2, creditCosts3, creditCosts4, creditCosts5,acc, auc, fnr))
perfRForest <- performance(rRForest$pred, measures = list(creditCosts1, creditCosts2, creditCosts3, creditCosts4, creditCosts5, acc, auc, fnr))
perfGBM <- performance(rGBM$pred, measures = list(creditCosts1, creditCosts2, creditCosts3, creditCosts4, creditCosts5, acc, auc, fnr))


# cost-sensitive empirical threshold tuning
ThLogitResult1 <- tuneThreshold(rLogit$pred, measure = creditCosts1, task = creditTask)
ThLogitResult2 <- tuneThreshold(rLogit$pred, measure = creditCosts2, task = creditTask)
ThLogitResult3 <- tuneThreshold(rLogit$pred, measure = creditCosts3, task = creditTask)
ThLogitResult4 <- tuneThreshold(rLogit$pred, measure = creditCosts4, task = creditTask)
ThLogitResult5 <- tuneThreshold(rLogit$pred, measure = creditCosts5, task = creditTask)

ThRForestResult1 <- tuneThreshold(rRForest$pred, measure = creditCosts1, task = creditTask)
ThRForestResult2 <- tuneThreshold(rRForest$pred, measure = creditCosts2, task = creditTask)
ThRForestResult3 <- tuneThreshold(rRForest$pred, measure = creditCosts3, task = creditTask)
ThRForestResult4 <- tuneThreshold(rRForest$pred, measure = creditCosts4, task = creditTask)
ThRForestResult5 <- tuneThreshold(rRForest$pred, measure = creditCosts5, task = creditTask)

ThGBMResult1 <- tuneThreshold(rGBM$pred, measure = creditCosts1, task = creditTask)
ThGBMResult2 <- tuneThreshold(rGBM$pred, measure = creditCosts2, task = creditTask)
ThGBMResult3 <- tuneThreshold(rGBM$pred, measure = creditCosts3, task = creditTask)
ThGBMResult4 <- tuneThreshold(rGBM$pred, measure = creditCosts4, task = creditTask)
ThGBMResult5 <- tuneThreshold(rGBM$pred, measure = creditCosts5, task = creditTask)


# extract empirical thresholds
ThLogit1 <- ThLogitResult1$th
ThLogit2 <- ThLogitResult2$th
ThLogit3 <- ThLogitResult3$th
ThLogit4 <- ThLogitResult4$th
ThLogit5 <- ThLogitResult5$th

ThRForest1 <- ThRForestResult1$th
ThRForest2 <- ThRForestResult2$th
ThRForest3 <- ThRForestResult3$th
ThRForest4 <- ThRForestResult4$th
ThRForest5 <- ThRForestResult5$th

ThGBM1 <- ThGBMResult1$th
ThGBM2 <- ThGBMResult2$th
ThGBM3 <- ThGBMResult3$th
ThGBM4 <- ThGBMResult4$th
ThGBM5 <- ThGBMResult5$th

# generating new predictions with the newly tuned (empirical) thresholds
TpredLogit1 <- setThreshold(rLogit$pred, ThLogit1)
TpredLogit2 <- setThreshold(rLogit$pred, ThLogit2)
TpredLogit3 <- setThreshold(rLogit$pred, ThLogit3)
TpredLogit4 <- setThreshold(rLogit$pred, ThLogit4)
TpredLogit5 <- setThreshold(rLogit$pred, ThLogit5)

TpredRForest1 <- setThreshold(rRForest$pred, ThRForest1)
TpredRForest2 <- setThreshold(rRForest$pred, ThRForest2)
TpredRForest3 <- setThreshold(rRForest$pred, ThRForest3)
TpredRForest4 <- setThreshold(rRForest$pred, ThRForest4)
TpredRForest5 <- setThreshold(rRForest$pred, ThRForest5)

TpredGBM1 <- setThreshold(rGBM$pred, ThGBM1)
TpredGBM2 <- setThreshold(rGBM$pred, ThGBM2)
TpredGBM3 <- setThreshold(rGBM$pred, ThGBM3)
TpredGBM4 <- setThreshold(rGBM$pred, ThGBM4)
TpredGBM5 <- setThreshold(rGBM$pred, ThGBM5)

# benchmarking the different learners by the cost measure with their respective empirical thresholds
TperfLogit1 <- performance(TpredLogit1, measures = list(creditCosts1, acc, auc, fnr))
TperfLogit2 <- performance(TpredLogit2, measures = list(creditCosts2, acc, auc, fnr))
TperfLogit3 <- performance(TpredLogit3, measures = list(creditCosts3, acc, auc, fnr))
TperfLogit4 <- performance(TpredLogit4, measures = list(creditCosts4, acc, auc, fnr))
TperfLogit5 <- performance(TpredLogit5, measures = list(creditCosts5, acc, auc, fnr))

TperfRForest1 <- performance(TpredRForest1, measures = list(creditCosts1, acc, auc, fnr))
TperfRForest2 <- performance(TpredRForest2, measures = list(creditCosts2, acc, auc, fnr))
TperfRForest3 <- performance(TpredRForest3, measures = list(creditCosts3, acc, auc, fnr))
TperfRForest4 <- performance(TpredRForest4, measures = list(creditCosts4, acc, auc, fnr))
TperfRForest5 <- performance(TpredRForest5, measures = list(creditCosts5, acc, auc, fnr))

TperfGBM1 <- performance(TpredGBM1, measures = list(creditCosts1, acc, auc, fnr))
TperfGBM2 <- performance(TpredGBM2, measures = list(creditCosts2, acc, auc, fnr))
TperfGBM3 <- performance(TpredGBM3, measures = list(creditCosts3, acc, auc, fnr))
TperfGBM4 <- performance(TpredGBM4, measures = list(creditCosts4, acc, auc, fnr))
TperfGBM5 <- performance(TpredGBM5, measures = list(creditCosts5, acc, auc, fnr))



#-----------------------------------------------------------------------#
############################## Tables ###################################
#-----------------------------------------------------------------------#
#Table 5.1
thresholdmatrix <- matrix(c(ThLogit1, ThLogit2, ThLogit3, ThLogit4, ThLogit5,
                            ThRForest1, ThRForest2, ThRForest3, ThRForest4, ThRForest5,
                            ThGBM1, ThGBM2, ThGBM3, ThGBM4, ThGBM5), ncol = 5, byrow = TRUE)
colnames(thresholdmatrix) <- c("CR 2:1", "CR 5:1", "CR 10:1", "CR 20:1", "CR 50:1")
rownames(thresholdmatrix) <- c("LR", "RF", "GB")
thresholdmatrix

#Table 5.2 calculated in Excel using the allcosts matrix with formula ((y2-y1)/y1)*100 and then multiplied by -1 because we only look at absolute percentage change. we know it's a reduction

#Table 5.3
modelperformance <- matrix(c(perfLogit[c(6:8)],
                             perfRForest[c(6:8)],
                             perfGBM[c(6:8)]), nrow = 3, byrow = TRUE)
colnames(modelperformance) <- names(perfLogit[c(6:8)])
rownames(modelperformance) <- c("LR", "RF", "GB")
modelperformance

#Figure 6.1
View(t(calculateConfusionMatrix(rGBM$pred)[1]$result))
View(t(calculateConfusionMatrix(TpredGBM3)[1]$result))
ThGBM3

## Appendix
allcosts <- matrix(c(perfLogit[1], TperfLogit1[1], perfLogit[2], TperfLogit2[1], perfLogit[3], TperfLogit3[1], perfLogit[4], TperfLogit4[1], perfLogit[5], TperfLogit5[1],
                     perfRForest[1], TperfRForest1[1], perfRForest[2], TperfRForest2[1], perfRForest[3], TperfRForest3[1], perfRForest[4], TperfRForest4[1], perfRForest[5], TperfRForest5[1],
                     perfGBM[1], TperfGBM1[1],  perfGBM[2], TperfGBM2[1], perfGBM[3], TperfGBM3[1], perfGBM[4], TperfGBM4[1], perfGBM[5],TperfGBM5[1] ), nrow = 3, byrow = TRUE)
colnames(allcosts) <- c("CPM1 0.5", "CPM1 tuned", "CPM2 0.5", "CPM2 tuned", "CPM3 0.5", "CPM3 tuned", "CPM4 0.5", "CPM4 tuned", "CPM5 0.5", "CPM5 tuned") # CPM means cost performance measure (= total average cost)
rownames(allcosts) <- c("LR", "RF", "GB")
allcosts

allaccuracies <- matrix(c(perfLogit[6], TperfLogit1[2], TperfLogit2[2], TperfLogit3[2], TperfLogit4[2], TperfLogit5[2],
                          perfRForest[6], TperfRForest1[2], TperfRForest2[2], TperfRForest3[2], TperfRForest4[2], TperfRForest5[2],
                          perfGBM[6], TperfGBM1[2], TperfGBM2[2], TperfGBM3[2], TperfGBM4[2], TperfGBM5[2]), nrow = 3, byrow = TRUE)
rownames(allaccuracies) <- c("LR", "RF", "GB")
colnames(allaccuracies) <- c("No CR", "2:1", "5:1", "10:1", "20:1", "50:1")



allfnrs <- matrix(c(perfLogit[8], TperfLogit1[4], TperfLogit2[4], TperfLogit3[4], TperfLogit4[4], TperfLogit5[4],
                    perfRForest[8], TperfRForest1[4], TperfRForest2[4], TperfRForest3[4], TperfRForest4[4], TperfRForest5[4],
                    perfGBM[8], TperfGBM1[4], TperfGBM2[4], TperfGBM3[4], TperfGBM4[4], TperfGBM5[4]), nrow = 3, byrow = TRUE)
rownames(allfnrs) <- c("LR", "RF", "GB")
colnames(allfnrs) <- c("No CR", "2:1", "5:1", "10:1", "20:1", "50:1")


allcosts # CPM means cost performance measure (= total average cost)
allaccuracies
allfnrs

# View(allcosts)
# View(allaccuracies)
# View(allfnrs)


#-----------------------------------------------------------------------#
############################### Plots ###################################
#-----------------------------------------------------------------------#
#Figure 5.1
# empirically tuned thresholds graph
tempNames <- c("CR 2:1", "CR 5:1", "CR 10:1", "CR 20:1", "CR 50:1")
tempNames <- factor(tempNames, levels = tempNames)
tempThsLR <- c(ThLogit1, ThLogit2, ThLogit3, ThLogit4, ThLogit5)
tempThsRF <- c(ThRForest1, ThRForest2, ThRForest3, ThRForest4, ThRForest5)
tempThsGB <- c(ThGBM1, ThGBM2, ThGBM3, ThGBM4, ThGBM5)
tempThsDf <- data.frame(tempThsLR, tempThsRF, tempThsGB)
tempThsDf %>% ggplot() +
  geom_point(aes(x=tempNames, y=tempThsLR), color="red") +
  geom_point(aes(x=tempNames, y=tempThsRF), color="blue") +
  geom_point(aes(x=tempNames, y=tempThsGB), color="green") +
  xlab("Cost Ratios") + ylab("Thresholds") 


#Figure 5.2
#801x433
Acc0 <- c(perfLogit[6], perfRForest[6], perfGBM[6])
Acc1 <- c(TperfLogit1[2], TperfRForest1[2], TperfGBM1[2])
Acc2 <- c(TperfLogit2[2], TperfRForest2[2], TperfGBM2[2])
Acc3 <- c(TperfLogit3[2], TperfRForest3[2], TperfGBM3[2])
Acc4 <- c(TperfLogit4[2], TperfRForest4[2], TperfGBM4[2])
Acc5 <- c(TperfLogit5[2], TperfRForest5[2], TperfGBM5[2])

tempNames <- c("Cost-Insensitive", "CR 2:1", "CR 5:1", "CR 10:1", "CR 20:1", "CR 50:1")
tempNames <- factor(tempNames, levels = tempNames)
tempACCsLR <- c(Acc0[1], Acc1[1], Acc2[1], Acc3[1], Acc4[1], Acc5[1])
tempACCsRF <- c(Acc0[2], Acc1[2], Acc2[2], Acc3[2], Acc4[2], Acc5[2])
tempACCsGB <- c(Acc0[3], Acc1[3], Acc2[3], Acc3[3], Acc4[3], Acc5[3])
tempACCsDf <- data.frame(tempACCsLR, tempACCsRF, tempACCsGB)
tempACCsDf %>% ggplot() +
  geom_point(aes(x=tempNames, y=tempACCsLR), color="red") +
  geom_point(aes(x=tempNames, y=tempACCsRF), color="blue") +
  geom_point(aes(x=tempNames, y=tempACCsGB), color="green") +
  xlab("Cost Ratios") + ylab("Accuracy")

#Figure 5.3
Fnr0 <- c(perfLogit[8], perfRForest[8], perfRForest[8])
Fnr1 <- c(TperfLogit1[4], TperfRForest1[4], TperfGBM1[4])
Fnr2 <- c(TperfLogit2[4], TperfRForest2[4], TperfGBM2[4])
Fnr3 <- c(TperfLogit3[4], TperfRForest3[4], TperfGBM3[4])
Fnr4 <- c(TperfLogit4[4], TperfRForest4[4], TperfGBM4[4])
Fnr5 <- c(TperfLogit5[4], TperfRForest5[4], TperfGBM5[4])

tempNames <- c("Cost-Insensitive", "CR 2:1", "CR 5:1", "CR 10:1", "CR 20:1", "CR 50:1")
tempNames <- factor(tempNames, levels = tempNames)
tempFNRsLR <- c(Fnr0[1], Fnr1[1], Fnr2[1], Fnr3[1], Fnr4[1], Fnr5[1])
tempFNRsRF <- c(Fnr0[2], Fnr1[2], Fnr2[2], Fnr3[2], Fnr4[2], Fnr5[2])
tempFNRsGB <- c(Fnr0[3], Fnr1[3], Fnr2[3], Fnr3[3], Fnr4[3], Fnr5[3])
tempFNRsDf <- data.frame(tempFNRsLR, tempFNRsRF, tempFNRsGB)
tempFNRsDf %>% ggplot() +
  geom_point(aes(x=tempNames, y=tempFNRsLR), color="red") +
  geom_point(aes(x=tempNames, y=tempFNRsRF), color="blue") +
  geom_point(aes(x=tempNames, y=tempFNRsGB), color="green") +
  xlab("Cost Ratios") + ylab("False Negative Rate")


#Figure 5.4
# threshold vs costs 3
thresholdDataGBM3 <- generateThreshVsPerfData(rGBM$pred, creditCosts3)
thresholdDataGBM3 <- data.frame("threshold" = thresholdDataGBM3$data$threshold, "creditCosts" = thresholdDataGBM3$data$creditCosts)
thresholdDataGBM3 %>% ggplot() +
  geom_line(aes(x = threshold, y = creditCosts), color = "navyblue") +
  geom_vline(xintercept = ThGBM3, color = "red") +
  geom_vline(xintercept = 0.5, color = "red", linetype = "dashed") +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1)) +
  theme(text = element_text(size = 12)) +
  xlab("Threshold") + ylab("Cost Measure")

# threshold vs accuracy
thresholdDataGBM30 <- generateThreshVsPerfData(rGBM$pred, acc)
thresholdDataGBM30 <- data.frame("threshold" = thresholdDataGBM30$data$threshold, "Accuracy" = thresholdDataGBM30$data$acc)
thresholdDataGBM30 %>% ggplot() +
  geom_line(aes(x = threshold, y = Accuracy), color = "navyblue") +
  geom_vline(xintercept = ThGBM3, color = "red") +
  geom_vline(xintercept = 0.5, color = "red", linetype = "dashed") +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1)) +
  theme(text = element_text(size = 12)) +
  xlab("Threshold") + ylab("Accuracy") 

# fnr vs thresholds gbm3
thresholdDataGBM31 <- generateThreshVsPerfData(rGBM$pred, fnr)
thresholdDataGBM31 <- data.frame("threshold" = thresholdDataGBM31$data$threshold, "FalseNegativeRate" = thresholdDataGBM31$data$fnr)
thresholdDataGBM31%>% ggplot() +
  geom_line(aes(x = threshold, y = FalseNegativeRate), color = "navyblue") +
  geom_vline(xintercept = ThGBM3, color = "red") +
  geom_vline(xintercept = 0.5, color = "red", linetype = "dashed") +
  scale_x_continuous(expand = c(0, 0), limits = c(0, 1)) +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1)) +
  theme(text = element_text(size = 12)) +
  xlab("Threshold") + ylab("False Negative Rate") 