#---------- Amit, Ankita, Darshan, Sagar-------
#----------Code was developed with similar to Naive Bayes explained in book(Machine Learning with R- Brett Lantz)---

install.packages("dplyr")
install.packages("tm")
install.packages("SnowballC")
install.packages("textmineR")
install.packages("wordcloud")
install.packages("e1071")
install.packages("gmodels")

library(gmodels)
library(e1071)
library(wordcloud)
library(SnowballC)
library(tm)
library(dplyr)
library(textmineR)
library(caret)

news_raw<-read.csv("Newsdata.csv", stringsAsFactors = FALSE)

prop.table(table(news_raw$Type))

news_raw$Type<-factor(news_raw$Type)
news_corpus<- VCorpus(VectorSource(news_raw$Content))

#------------DTM----------

newsdtm2<-DocumentTermMatrix(news_corpus, control = list(tolower = TRUE, 
                                                         removeNumbers = TRUE, 
                                                         stopwords = TRUE, 
                                                         removePunctuation = TRUE, stemDocument = TRUE))


news_dtm_train<-newsdtm2[1:1500,]
news_dtm_test<-newsdtm2[1501:2000,]

news_train_labels<-news_raw[1:1500,]$Type
news_test_labels<-news_raw[1501:2000,]$Type

prop.table(table(news_train_labels))
prop.table(table(news_test_labels))

#--------frequent words---------
news_freq_words<-findFreqTerms(news_dtm_train,5)
news_dtm_freq_train<- news_dtm_train[,news_freq_words]
news_dtm_freq_test<- news_dtm_test[,news_freq_words]

##-----------converting numeric values to categorical value---
convert_counts<-function(x){
  x<-ifelse(x>0, "Yes","No")
}

news_train<-apply(news_dtm_freq_train, MARGIN = 2, convert_counts)
news_test<-apply(news_dtm_freq_test, MARGIN = 2, convert_counts)

#---------Applying Naive bayes------------
news_classifier<-naiveBayes(news_train,news_train_labels)
news_test_pred<-predict(news_classifier, news_test,type= "class")

#---------Checking Performance---------------------
CrossTable(news_test_pred, news_test_labels,prop.chisq = FALSE,prop.t = FALSE,dnn = c('predicted','actual'))
caret::confusionMatrix(news_test_pred,news_test_labels, positive = "Yes")
