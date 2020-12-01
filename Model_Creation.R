########################################################### loading packages ########################################################################
library(twitteR)
library(ROAuth)
library(tidyverse)
library(text2vec)
library(caret)
library(glmnet)
library(ggrepel)
library(purrrlyr)


############################################### function for converting some symbols ################################################################
conv_fun <- function(x) iconv(x, "latin1", "ASCII", "")



########################################################## load dataset #############################################################################
tweets_classified <- read_csv('training.1600000.processed.noemoticon.csv',
                              col_names = c('sentiment', 'id', 'date', 'query', 'user', 'text')) %>%
  # converting some symbols
  dmap_at('text', conv_fun) %>%
  # replacing class values
  mutate(sentiment = ifelse(sentiment == 0, 0, 1))



################################################ Overall Idea about sentiment entries ###############################################################
prop.table(table(tweets_classified$sentiment))



################################################# data splitting on train and test ##################################################################
set.seed(2340)
trainIndex <- createDataPartition(tweets_classified$sentiment, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
tweets_train <- tweets_classified[trainIndex, ]
tweets_test <- tweets_classified[-trainIndex, ]



############################################################# doc2vec ###############################################################################
# define preprocessing function and tokenization function
prep_fun <- tolower
tok_fun <- word_tokenizer

it_train <- itoken(tweets_train$text, 
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun,
                   ids = tweets_train$id,
                   progressbar = TRUE)
it_test <- itoken(tweets_test$text, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun,
                  ids = tweets_test$id,
                  progressbar = TRUE)



################################################ creating vocabulary and document-term matrix #######################################################
vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)
dtm_test <- create_dtm(it_test, vectorizer)



############################################################## define tf-idf model ##################################################################
tfidf <- TfIdf$new()



####################################### fit the model to the train data and transform it with the fitted model #####################################
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
dtm_test_tfidf <- fit_transform(dtm_test, tfidf)
getModelInfo()$glmnet$type




################################################################### train the model #################################################################
t1 <- Sys.time()
glmnet_classifier <- cv.glmnet(x = dtm_train_tfidf, y = tweets_train[['sentiment']], 
                               family = 'binomial', 
                               # L1 penalty
                               alpha = 1,
                               # interested in the area under ROC curve
                               type.measure = "auc",
                               # 5-fold cross-validation
                               nfolds = 5,
                               # high value is less accurate, but has faster training
                               thresh = 1e-3,
                               # again lower number of iterations for faster training
                               maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'mins'))



plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

preds <- predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[ ,1]
glmnet:::auc(as.numeric(tweets_test$sentiment), preds)




############################################################### save the model ######################################################################
saveRDS(glmnet_classifier, 'TFIDF.RDS')
