########################################################### loading packages ########################################################################
library(twitteR)
library(ROAuth)
library(tidyverse)
library(text2vec)
library(caret)
library(glmnet)
library(ggrepel)
library(purrrlyr)

#################################### Connecting & authentication of Twitter developer account ########################################################

download.file(url = "http://curl.haxx.se/ca/cacert.pem",
              destfile = "cacert.pem")
setup_twitter_oauth('XXXXXXXXXXXXXXXXXXXXX', # api key
                    'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', # api secret
                    'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', # access token
                    'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX' # access token secret
)




############################################################ fetching tweets #########################################################################
df_tweets <- twListToDF(searchTwitter('happy', n = 1000, lang = 'en')) %>%
  #converting some symbols
  dmap_at('text', conv_fun)




################################################## preprocessing and tokenization ####################################################################
it_tweets <- itoken(df_tweets$text,
                    preprocessor = prep_fun,
                    tokenizer = tok_fun,
                    ids = df_tweets$id,
                    progressbar = TRUE)




############################################# creating vocabulary and document-term matrix ###########################################################
dtm_tweets <- create_dtm(it_tweets, vectorizer)




################################################## transforming data with tf-idf #####################################################################
dtm_tweets_tfidf <- fit_transform(dtm_tweets, tfidf)




################################################### loading classification model #####################################################################
glmnet_classifier <- readRDS('TFIDF.RDS')




################################################ predict probabilities of positiveness ###############################################################
preds_tweets <- predict(glmnet_classifier, dtm_tweets_tfidf, type = 'response')[ ,1]




################################################## adding rates to initial dataset ###################################################################
df_tweets$sentiment <- preds_tweets


