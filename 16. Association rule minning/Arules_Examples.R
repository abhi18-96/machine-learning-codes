library(arules)

data(Groceries)
class(Groceries)
isS4(Groceries)
slotNames(Groceries)

# Groceries@transactionInfo
# A data frame with vectors of same length as number of transactions

Groceries@data
# A binary incidence matrix that indicates which item labels appear
# in every transaction

Groceries@itemInfo
# A data frame to store item labels

Groceries@itemInfo[1:15,]

itemsets <- apriori(Groceries,
                    parameter = list(minlen=1, maxlen=1,
                                     support=0.05,
                                     target="frequent itemsets"))

#summary(itemsets)

itemFrequencyPlot(Groceries,topN=10)

inspect(head(sort(itemsets, by="support"),15))

inspect(sort(itemsets, by="support"))


# minlen and maxlen changed

itemsets <- apriori(Groceries, parameter = list(minlen=1, maxlen=2,
                                                support=0.05,
                                                target="frequent itemsets"))
inspect(head(sort(itemsets, by="support"),20))


itemsets <- apriori(Groceries,
                    parameter = list(minlen=2, maxlen=2,
                                                support=0.02,
                                     target="frequent itemsets"))
inspect(head(sort(itemsets, by="support"),10))

# Rules Generation
rules <- apriori(Groceries,
                 parameter = list(   support=0.001,
                                  confidence=0.8,
                                      target="rules"),
                 control = list( verbose = F))

inspect(head(sort(rules,by="lift"),10))

inspect(head(sort(rules,by=c("confidence"))))

summary(rules)

# Visualizing Rules

library(arulesViz)

plot(rules)

head(rules@quality)

plot(rules@quality)

head(quality(rules))

highLiftRules <- head(sort(rules,by="lift"),5)
inspect(highLiftRules)
plot(highLiftRules,
     method="graph",control=list(type="items"),
     engine = "htmlwidget")

plot(highLiftRules,
     method="graph",control=list(type="items"))


###########################################################################
confidentRules <- apriori(Groceries,
                          parameter = list(   support=0.001,
                                              confidence=0.9,
                                              target="rules"),
                          control = list( verbose = F))

# confidentRules <- rules[quality(rules)$confidence > 0.9
#                         & quality(rules)$support > 0.001
#                         & quality(rules)$lift > 1.5]

inspect(head(sort(confidentRules,by="lift"),5))


highLiftRules <- head(sort(confidentRules,by="lift"),5)
plot(highLiftRules,
     method="graph",control=list(type="items"),
     engine = "htmlwidget")
inspect(highLiftRules)

inspect(head(sort(rules,by="lift"),5))

#### Parallel Coordinates Plot #####
plot(highLiftRules, method="paracoord")
