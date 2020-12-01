########################################################### loading packages ########################################################################
install.packages("Metrics")
install.packages("ROCR")

library(twitteR)
library(ROAuth)
library(tidyverse)
library(text2vec)
library(caret)
library(glmnet)
library(ggrepel)
library(purrrlyr)
library(Metrics)
library(ROCR)
library(pROC)
library(nnet)


###################################################### preprocessing and tokenization ################################################################
t2 <- Sys.time()
df_tweets <- tweets_test
it_tweets <- itoken(df_tweets$text,
                    preprocessor = prep_fun,
                    tokenizer = tok_fun,
                    ids = df_tweets$id,
                    progressbar = TRUE)




################################################# creating vocabulary and document-term matrix #######################################################
dtm_tweets <- create_dtm(it_tweets, vectorizer)




####################################################### transforming data with tf-idf ################################################################
dtm_tweets_tfidf <- fit_transform(dtm_tweets, tfidf)




####################################################### loading classification model #################################################################
glmnet_classifier <- readRDS('TFIDF.RDS')




#################################################### predict probabilities of positiveness ###########################################################

preds_tweets <- predict(glmnet_classifier, dtm_tweets_tfidf, type = 'response')[ ,1]
preds_tweets$sentiment <- preds_tweets
t3 <- Sys.time()




####################################################### adding rates to initial dataset ##############################################################
df_tweets$sentiment <- ifelse(preds_tweets$sentiment > 0.50,
                                          1,
                                          0)


print(t3-t2)

############################################################# Confusion Matrix #######################################################################
tab <- table(df_tweets$sentiment, tweets_test$sentiment)
tab




################################################################ Accuracy ############################################################################
Accuracy <- accuracy(df_tweets$sentiment,tweets_test$sentiment)
Accuracy




################################################################## AUC ###############################################################################
AUC <- Metrics::auc(df_tweets$sentiment,tweets_test$sentiment)
AUC




################################################################ ROC plot ############################################################################
data(ROCR.simple)
df <- data.frame(ROCR.simple)
pred <- prediction(preds_tweets$sentiment, tweets_test$sentiment)
perf <- performance(pred,"tpr","fpr")
plot(perf,colorize=TRUE)
abline(a=0, b= 1)
