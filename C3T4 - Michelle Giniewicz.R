# Title: C3T4 - Discover Associations Between Products

#Updated:  8/2/2022


###############
# Project Notes
###############


# Clear console: CTRL + L


###############
# Housekeeping
###############

# Clear objects if necessary
rm(list = ls())

# get working directory
getwd()

# set working directory 
setwd("C:/Users/giniewic/OneDrive - HP Inc/Documents/Personal/UT Data Analytics Cert/Course 3/C3T4")
dir()



###############
# Load packages
###############
install.packages("Rtools")
install.packages("caret")
install.packages("corrplot")
install.packages("readr")
install.packages("mlbench")
install.packages("doParallel")
install.packages("reshape2")
install.packages("dplyr")
install.packages("arules")
install.packages("arulesViz")
library(caret)
library(corrplot)
library(readr)
library(mlbench)
library(doParallel)
library(e1071)
library(gbm)
library(ggplot2)
library(writexl)
library(reshape2)
library(dplyr)
library(arules)
library(arulesViz)




#####################
# Parallel Processing
#####################

#detectCores()         #detect number of cores
#cl <- makeCluster(2)  # select number of cores
#registerDoParallel(cl) # register cluster
#getDoParWorkers()      # confirm number of cores being used by RStudio
#  Stop Cluster -- After performing tasks, make sure to stop cluster
#stopCluster(cl)
#detectCores()


####################
# Import data
####################

# For transaction data, need to use read.transactions() function
##-- Load the dataset
TransactionOOB <- read.transactions("ElectronidexTransactions2017.csv", format="basket", sep=",", rm.duplicates=TRUE)
str(TransactionOOB)


######################
# Save datasets
######################




##################
# Evaluate data
##################

# View first/last obs/rows
#head(TransactionOOB)
#tail(TransactionOOB)
#anyNA(TransactionOOB)
#anyDuplicated(TransactionOOB)
summary(TransactionOOB)

inspect(TransactionOOB) #View transactions
inspect(TransactionOOB[1:10])  #view top 10 transactions
length(TransactionOOB)  # Number of transactions
size(TransactionOOB)  # Number of items per transaction
size(TransactionOOB[1:10])  # Number of items per transaction for top 10
LIST(TransactionOOB)  # Lists transactions by conversion
LIST(TransactionOOB[1:10])  # Lists transactions by conversion for top 10

itemLabels(TransactionOOB)  # See item labels


### -- OBSERVATIONS -- ###
# There are 125 possible items
# There are 9835 transactions


#############
# Preprocess
#############


#####################
# EDA/Visualizations
#####################

#--- Statistics ---#
summary(TransactionOOB)

#transactions as itemMatrix in sparse format with
#9835 rows (elements/itemsets/transactions) and
#125 columns (items) and a density of 0.03506172 

#most frequent items:
#  iMac                HP Laptop CYBERPOWER Gamer Desktop 
#2519                     1909                     1809 
#Apple Earpods        Apple MacBook Air                  (Other) 
#1715                     1530                    33622 


#--- Plots ---#
# Plot top 20 items
itemFrequencyPlot(TransactionOOB, topN=20, type='absolute', main="Top 20 Items")
# Plot top 10 items
itemFrequencyPlot(TransactionOOB, topN=10, type='absolute', main="Top 10 Items (absolute)")
itemFrequencyPlot(TransactionOOB, topN=10, type='relative', main="Top 10 Items (relative)")

barplot(sort(itemFrequency(TransactionOOB), decreasing=FALSE))


# Visualize transactions in dataset using image()
image(TransactionOOB,main="Transactions")
image(sample(TransactionOOB, 50), main="50 Transactions")
image(sample(TransactionOOB, 100), main="100 Transactions")
image(sample(TransactionOOB, 25), main="25 Transactions")

# Transaction visualization seems randomly distributed - nothing big stands out


##################
# Train/test sets
##################

# We do not need to train because Apriori Algorithm is an unsupervised learner


#####################
# Modeling
#####################

###--- Apriori Algorithm ---###
#The Apriori algorithm is helpful when working with large datasets and is used 
#to uncover insights pertaining to transactional datasets. It is based on item 
#frequency. For example, this item set {Item 1, Item 2, Item 3, Item 4} can only 
#occur if items {Item 1}, {Item 2}, {Item 3} and {Item 4} occur just as frequently.
## Support - measures itemsets/rules frequency
## Confidence - measures accuracy of the rules
# *** Strong Rule = rules measures high in both Support and Confidence *** #

# Find association rules - OOB
OOBRules <- apriori(TransactionOOB, parameter=list(supp=0.1, conf=0.8))

# View rules - OOB
summary(OOBRules)

## --- 0 rules created - need to experiment with values

# Find association rules
Rules2 <- apriori(TransactionOOB, parameter=list(supp=0.5, conf=0.8))
summary(Rules2)
# still 0 rules

Rules3 <- apriori(TransactionOOB, parameter=list(supp=0.05, conf=0.8))
summary(Rules3)
# still 0 rules

Rules4 <- apriori(TransactionOOB, parameter=list(supp=0.001, conf=0.8))
summary(Rules4)
# Number of rules: 635
# Dist by length: length of 4 has the most rules
inspect(Rules4[1:10])
# 90.9% of customers who bought ASUS 2 monitor, Generic Black 3-Button also bought iMac
# 90.9% of customers who bough 3-Button Mouse, Fire TV Stick also bough iMac

Rules5 <- apriori(TransactionOOB, parameter=list(supp=0.001, conf=0.9))
summary(Rules5)
# Number of rules: 197
# Dist by length: length of 5 has the most rules
inspect(Rules5[1:15])
# 100% of customers who bought Brother Printer, Halter Acrylic Monitor Stand also bought iMac
# 100% of customers who bought ASUS Monitor, Mackie CR Speakers, ViewSonic Monitor also bought iMac

### Minlen = minimum items required in a rule
Rules6 <- apriori(TransactionOOB, parameter=list(supp=0.001, conf=0.9, minlen=2))
summary(Rules6)
# Number of rules: 197
# Dist by length: length of 5 has the most rules
inspect(Rules6[1:15])

##################
# Improve Model
##################

#Sort rules by confidence
RulesSorted <- sort(Rules5, decreasing=TRUE, na.last=NA, by="confidence", order=FALSE)

inspect(RulesSorted)

# the first 42 rules have confidence of 1 - only view those
inspect(RulesSorted[1:42])

# Sort rules with only 100% confidence, by lift
RulesSorted1 <- sort(RulesSorted[1:42], decreasing=TRUE, na.last=NA, by="lift", order=FALSE)

inspect(RulesSorted1)
#[1]  {Dell Desktop, iMac, Lenovo Desktop Computer, Mackie CR Speakers}                                          => {ViewSonic Monitor}       0.001118454 1          0.001118454 9.064516 11   
#[2]  {Dell Desktop, HP Laptop, iMac, Lenovo Desktop Computer, Mackie CR Speakers}                               => {ViewSonic Monitor}       0.001016777 1          0.001016777 9.064516 10   
#[3]  {ASUS Monitor, Intel Desktop, ViewSonic Monitor}                                                           => {Lenovo Desktop Computer} 0.001118454 1          0.001118454 6.754808 11   
#[4]  {ASUS Monitor, iMac, Intel Desktop, ViewSonic Monitor}                                                     => {Lenovo Desktop Computer} 0.001016777 1          0.001016777 6.754808 10   
#[5]  {Acer Aspire, Koss Home Headphones, ViewSonic Monitor}                                                     => {HP Laptop}               0.001220132 1          0.001220132 5.151912 12   
#[6]  {Dell Desktop, Koss Home Headphones, ViewSonic Monitor}                                                    => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[7]  {Acer Aspire, ASUS 2 Monitor, Intel Desktop}                                                               => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[8]  {ASUS 2 Monitor, Computer Game, Dell Desktop}                                                              => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[9]  {Dell Desktop, iMac, Lenovo Desktop Computer, Logitech MK270 Wireless Keyboard and Mouse Combo}            => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[10] {Acer Desktop, Dell Desktop, HDMI Cable 6ft, iMac}                                                         => {HP Laptop}               0.001321810 1          0.001321810 5.151912 13   
#[11] {Acer Aspire, Computer Game, Dell Desktop, ViewSonic Monitor}                                              => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[12] {Apple Magic Keyboard, ASUS Monitor, Dell Desktop, HP Monitor}                                             => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[13] {Apple Earpods, Backlit LED Gaming Keyboard, CYBERPOWER Gamer Desktop, iMac}                               => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[14] {Acer Desktop, Apple Magic Keyboard, Dell Desktop, HP Monitor}                                             => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[15] {Acer Aspire, Apple Magic Keyboard, Dell Desktop, ViewSonic Monitor}                                       => {HP Laptop}               0.001423488 1          0.001423488 5.151912 14   
#[16] {3-Button Mouse, Acer Aspire, Dell Desktop, ViewSonic Monitor}                                             => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[17] {Acer Aspire, Apple Magic Keyboard, Dell Desktop, iMac, ViewSonic Monitor}                                 => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[18] {Brother Printer, Halter Acrylic Monitor Stand}                                                            => {iMac}                    0.001118454 1          0.001118454 3.904327 11   
#[19] {ASUS Monitor, Mackie CR Speakers, ViewSonic Monitor}                                                      => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[20] {Apple Magic Keyboard, Rii LED Gaming Keyboard & Mouse Combo, ViewSonic Monitor}                           => {iMac}                    0.001728521 1          0.001728521 3.904327 17   
#[21] {ASUS Monitor, Koss Home Headphones, Microsoft Office Home and Student 2016}                               => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[22] {ASUS 2 Monitor, Dell Desktop, Logitech Keyboard}                                                          => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[23] {Alienware Laptop, ASUS Desktop, Lenovo Desktop Computer}                                                  => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[24] {Brother Printer, Dell Desktop, Epson Printer}                                                             => {iMac}                    0.001118454 1          0.001118454 3.904327 11   
#[25] {Apple Magic Keyboard, Brother Printer, ViewSonic Monitor}                                                 => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[26] {ASUS Desktop, Dell Desktop, Microsoft Office Home and Student 2016}                                       => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[27] {Intel Desktop, iPad Pro, Microsoft Office Home and Student 2016}                                          => {iMac}                    0.001118454 1          0.001118454 3.904327 11   
#[28] {Acer Aspire, Apple MacBook Pro, HP Black & Tri-color Ink, HP Laptop}                                      => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[29] {Dell Desktop, Etekcity Power Extension Cord Cable, Lenovo Desktop Computer, ViewSonic Monitor}            => {iMac}                    0.001220132 1          0.001220132 3.904327 12   
#[30] {Etekcity Power Extension Cord Cable, HP Laptop, Lenovo Desktop Computer, ViewSonic Monitor}               => {iMac}                    0.001626843 1          0.001626843 3.904327 16   
#[31] {Apple Magic Keyboard, ASUS Desktop, Dell Desktop, Lenovo Desktop Computer}                                => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[32] {AOC Monitor, ASUS Monitor, HP Laptop, ViewSonic Monitor}                                                  => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[33] {AOC Monitor, Dell Desktop, Lenovo Desktop Computer, ViewSonic Monitor}                                    => {iMac}                    0.001118454 1          0.001118454 3.904327 11   
#[34] {Epson Printer, HP Monitor, Lenovo Desktop Computer, ViewSonic Monitor}                                    => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[35] {Apple Magic Keyboard, ASUS Monitor, HP Laptop, LG Monitor}                                                => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[36] {Apple Magic Keyboard, ASUS Monitor, Dell Desktop, Microsoft Office Home and Student 2016}                 => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[37] {Apple Magic Keyboard, ASUS Monitor, HP Laptop, Microsoft Office Home and Student 2016}                    => {iMac}                    0.001220132 1          0.001220132 3.904327 12   
#[38] {Acer Desktop, Apple Magic Keyboard, ASUS Monitor, ViewSonic Monitor}                                      => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[39] {3-Button Mouse, Acer Aspire, Apple Magic Keyboard, CYBERPOWER Gamer Desktop}                              => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[40] {CYBERPOWER Gamer Desktop, Dell Desktop, Samsung Monitor, ViewSonic Monitor}                               => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[41] {Dell Desktop, Etekcity Power Extension Cord Cable, HP Laptop, Lenovo Desktop Computer, ViewSonic Monitor} => {iMac}                    0.001118454 1          0.001118454 3.904327 11   
#[42] {Acer Desktop, HP Laptop, HP Monitor, Lenovo Desktop Computer, ViewSonic Monitor}                          => {iMac}                    0.001118454 1          0.001118454 3.904327 11

# Look at only iMac rules to see if that is helpful
iMacRules <- subset(Rules5, items %in% "iMac")
summary(iMacRules)
#148 rules have iMac in them

#view iMac rules sorted by confidence
inspect(sort(iMacRules, decreasing=TRUE, na.last=NA, by="confidence", order=FALSE))

# Check for redundant rules
is.redundant(Rules5)

# Remove redundant rules 
Rules5NoRedundant <- Rules5[!is.redundant(Rules5)]

# check that redundant rules were removed
is.redundant(Rules5NoRedundant)

# Sort rules with only 100% confidence, by lift
#Sort rules by confidence
RulesSortedNoRedundant <- sort(Rules5NoRedundant, decreasing=TRUE, na.last=NA, by="confidence", order=FALSE)
inspect(RulesSortedNoRedundant)

# the first 38 rules have confidence of 1 - only view those
inspect(RulesSortedNoRedundant[1:38])

# Sort rules with only 100% confidence, by lift
RulesSortedNoRedundant1 <- sort(RulesSortedNoRedundant[1:38], decreasing=TRUE, na.last=NA, by="lift", order=FALSE)
inspect(RulesSortedNoRedundant1)

#     lhs                                                                                                rhs                       support     confidence coverage    lift     count
#[1]  {Dell Desktop, iMac, Lenovo Desktop Computer, Mackie CR Speakers}                               => {ViewSonic Monitor}       0.001118454 1          0.001118454 9.064516 11   
#[2]  {ASUS Monitor, Intel Desktop, ViewSonic Monitor}                                                => {Lenovo Desktop Computer} 0.001118454 1          0.001118454 6.754808 11   
#[3]  {Acer Aspire, Koss Home Headphones, ViewSonic Monitor}                                          => {HP Laptop}               0.001220132 1          0.001220132 5.151912 12   
#[4]  {Dell Desktop, Koss Home Headphones, ViewSonic Monitor}                                         => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[5]  {Acer Aspire, ASUS 2 Monitor, Intel Desktop}                                                    => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[6]  {ASUS 2 Monitor, Computer Game, Dell Desktop}                                                   => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[7]  {Dell Desktop, iMac, Lenovo Desktop Computer, Logitech MK270 Wireless Keyboard and Mouse Combo} => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[8]  {Acer Desktop, Dell Desktop, HDMI Cable 6ft, iMac}                                              => {HP Laptop}               0.001321810 1          0.001321810 5.151912 13   
#[9]  {Acer Aspire, Computer Game, Dell Desktop, ViewSonic Monitor}                                   => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[10] {Apple Magic Keyboard, ASUS Monitor, Dell Desktop, HP Monitor}                                  => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[11] {Apple Earpods, Backlit LED Gaming Keyboard, CYBERPOWER Gamer Desktop, iMac}                    => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[12] {Acer Desktop, Apple Magic Keyboard, Dell Desktop, HP Monitor}                                  => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[13] {Acer Aspire, Apple Magic Keyboard, Dell Desktop, ViewSonic Monitor}                            => {HP Laptop}               0.001423488 1          0.001423488 5.151912 14   
#[14] {3-Button Mouse, Acer Aspire, Dell Desktop, ViewSonic Monitor}                                  => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[15] {Brother Printer, Halter Acrylic Monitor Stand}                                                 => {iMac}                    0.001118454 1          0.001118454 3.904327 11   
#[16] {ASUS Monitor, Mackie CR Speakers, ViewSonic Monitor}                                           => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[17] {Apple Magic Keyboard, Rii LED Gaming Keyboard & Mouse Combo, ViewSonic Monitor}                => {iMac}                    0.001728521 1          0.001728521 3.904327 17   
#[18] {ASUS Monitor, Koss Home Headphones, Microsoft Office Home and Student 2016}                    => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[19] {ASUS 2 Monitor, Dell Desktop, Logitech Keyboard}                                               => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[20] {Alienware Laptop, ASUS Desktop, Lenovo Desktop Computer}                                       => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[21] {Brother Printer, Dell Desktop, Epson Printer}                                                  => {iMac}                    0.001118454 1          0.001118454 3.904327 11   
#[22] {Apple Magic Keyboard, Brother Printer, ViewSonic Monitor}                                      => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[23] {ASUS Desktop, Dell Desktop, Microsoft Office Home and Student 2016}                            => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[24] {Intel Desktop, iPad Pro, Microsoft Office Home and Student 2016}                               => {iMac}                    0.001118454 1          0.001118454 3.904327 11   
#[25] {Acer Aspire, Apple MacBook Pro, HP Black & Tri-color Ink, HP Laptop}                           => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[26] {Dell Desktop, Etekcity Power Extension Cord Cable, Lenovo Desktop Computer, ViewSonic Monitor} => {iMac}                    0.001220132 1          0.001220132 3.904327 12   
#[27] {Etekcity Power Extension Cord Cable, HP Laptop, Lenovo Desktop Computer, ViewSonic Monitor}    => {iMac}                    0.001626843 1          0.001626843 3.904327 16   
#[28] {Apple Magic Keyboard, ASUS Desktop, Dell Desktop, Lenovo Desktop Computer}                     => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[29] {AOC Monitor, ASUS Monitor, HP Laptop, ViewSonic Monitor}                                       => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[30] {AOC Monitor, Dell Desktop, Lenovo Desktop Computer, ViewSonic Monitor}                         => {iMac}                    0.001118454 1          0.001118454 3.904327 11   
#[31] {Epson Printer, HP Monitor, Lenovo Desktop Computer, ViewSonic Monitor}                         => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[32] {Apple Magic Keyboard, ASUS Monitor, HP Laptop, LG Monitor}                                     => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[33] {Apple Magic Keyboard, ASUS Monitor, Dell Desktop, Microsoft Office Home and Student 2016}      => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[34] {Apple Magic Keyboard, ASUS Monitor, HP Laptop, Microsoft Office Home and Student 2016}         => {iMac}                    0.001220132 1          0.001220132 3.904327 12   
#[35] {Acer Desktop, Apple Magic Keyboard, ASUS Monitor, ViewSonic Monitor}                           => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[36] {3-Button Mouse, Acer Aspire, Apple Magic Keyboard, CYBERPOWER Gamer Desktop}                   => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[37] {CYBERPOWER Gamer Desktop, Dell Desktop, Samsung Monitor, ViewSonic Monitor}                    => {iMac}                    0.001016777 1          0.001016777 3.904327 10   
#[38] {Acer Desktop, HP Laptop, HP Monitor, Lenovo Desktop Computer, ViewSonic Monitor}               => {iMac}                    0.001118454 1          0.001118454 3.904327 11   

# Get only rules with Lift >5 (top 14 rules)
SubsetRulesSorted <- RulesSortedNoRedundant1[1:14]
inspect(SubsetRulesSorted)

#     lhs                                                                                                rhs                       support     confidence coverage    lift     count
#[1]  {Dell Desktop, iMac, Lenovo Desktop Computer, Mackie CR Speakers}                               => {ViewSonic Monitor}       0.001118454 1          0.001118454 9.064516 11   
#[2]  {ASUS Monitor, Intel Desktop, ViewSonic Monitor}                                                => {Lenovo Desktop Computer} 0.001118454 1          0.001118454 6.754808 11   
#[3]  {Acer Aspire, Koss Home Headphones, ViewSonic Monitor}                                          => {HP Laptop}               0.001220132 1          0.001220132 5.151912 12   
#[4]  {Dell Desktop, Koss Home Headphones, ViewSonic Monitor}                                         => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[5]  {Acer Aspire, ASUS 2 Monitor, Intel Desktop}                                                    => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[6]  {ASUS 2 Monitor, Computer Game, Dell Desktop}                                                   => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[7]  {Dell Desktop, iMac, Lenovo Desktop Computer, Logitech MK270 Wireless Keyboard and Mouse Combo} => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[8]  {Acer Desktop, Dell Desktop, HDMI Cable 6ft, iMac}                                              => {HP Laptop}               0.001321810 1          0.001321810 5.151912 13   
#[9]  {Acer Aspire, Computer Game, Dell Desktop, ViewSonic Monitor}                                   => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[10] {Apple Magic Keyboard, ASUS Monitor, Dell Desktop, HP Monitor}                                  => {HP Laptop}               0.001118454 1          0.001118454 5.151912 11   
#[11] {Apple Earpods, Backlit LED Gaming Keyboard, CYBERPOWER Gamer Desktop, iMac}                    => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[12] {Acer Desktop, Apple Magic Keyboard, Dell Desktop, HP Monitor}                                  => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   
#[13] {Acer Aspire, Apple Magic Keyboard, Dell Desktop, ViewSonic Monitor}                            => {HP Laptop}               0.001423488 1          0.001423488 5.151912 14   
#[14] {3-Button Mouse, Acer Aspire, Dell Desktop, ViewSonic Monitor}                                  => {HP Laptop}               0.001016777 1          0.001016777 5.151912 10   

#########################
# Visualize Model Results
#########################

# Plot rules as scatterplot
plot(SubsetRulesSorted)

# Plot graph
plot(SubsetRulesSorted, method="graph", control=list(type="items"))

plot(SubsetRulesSorted[1:10], method="graph", control=list(type="items"), main="Graph of Top 10 Rules")

plot(SubsetRulesSorted[1:5], method="graph", control=list(type="items"), main="Graph of Top 5 Rules")


# Other plots  ---- Available methods: ‘matrix’, ‘mosaic’, ‘doubledecker’, ‘graph’, ‘paracoord’, ‘scatterplot’, ‘grouped matrix’, ‘two-key plot’, ‘matrix3D’
plot(SubsetRulesSorted, method="paracoord", control=list(type="items"))

plot(SubsetRulesSorted[1:10], method="paracoord", control=list(type="items"))

plot(SubsetRulesSorted, method="matrix", control=list(type="items"))

plot(SubsetRulesSorted[1:10], method="matrix", control=list(type="items"))

plot(SubsetRulesSorted, method="grouped matrix", control=list(type="items"))

plot(SubsetRulesSorted, method="two-key plot", control=list(type="items"))

plot(SubsetRulesSorted, method="matrix3D", control=list(type="items"))

