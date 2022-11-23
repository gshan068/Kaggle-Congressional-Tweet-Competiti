[congressional_tweets_project_report.pdf](https://github.com/gshan068/Kaggle-Congressional-Tweet-Competiti/files/10072140/congressional_tweets_project_report.pdf)
# Kaggle-Congressional-Tweet-Competition
INTRODUCTION:
This tweet dataset is a collection of tweets posted by Congressional politicians on Twitter. All tweets here are excluding tweets that have no hashtags. We used the training dataset to build models and make some predictions on the testing dataset. A logistic regression method is the process of modeling the probability of discrete outcomes given input variables. It estimates the parameters of a logistic model. There are many different types of logistic regressions, such as binary logistic regression, multinomial logistic regression, and ordinal logistic regression. However, here we used bigram logistic regression by implementing the LogisticRegressionCV because it can help select the regularization coefficient. The rest of this report is structured as follows. Several descriptive findings include some tables and visuals after exploring the dataset and applying logistic regression models. Finally, we came up to about 0.89247 accuracy score.

A. Findings
• The democrats party’s tweets are more about topics such as climate change, president Trump, and census. However, the republican’s tweets are related to topics about President Obama, tax reform, jobs, and so on.
• The ratio of the republican and democrat party for a total of 265000 tweets is about 50:50, which may imply that there are no big biases.
• Some of the words that have bigger coefficients related to the party are something such as gop tax scam (-17.886353), trump shut down(-16.748860), tcot(21.265651), and so on.


DESCRIPTIVE ANALYSIS: 
A. Use of hashtags and hashtags patterns comparing Repub- licans and Democrats:
The democrats’ tweets are related to the census, climate change, president Trump, and so on. However, the republican’ tweets are related to topics about President Obama, tax reform, and jobs. There are also some hashtags that cannot really predict a party, such as covid19 and coronavirus, because the are included in both parties’ frequent item-sets.
We applied association rule into hashtags columns. We chose the minimum support 0.005 to show all frequent item-sets for each party by using Apriori
After applying the logistic regression model, we checked for the coefficients or weights of each hashtag content. Some of the words that have bigger coefficients related to the party, such as gop tax scam, trump shut down, defend our democracy, and so on.
B. Polarity Analysis
Polarity means emotions expressed in a sentence. It lies in the range of [-1,1], where 1 means positive statement and -1 means a negative statement. For the full textsup and hashtags combined, we saw that the polarity mainly ranges between 0.00 and 0.50. This indicates that the majority of the words in tweets are positive.


CLASSIFICATION METHODS:
A. Data-Preprocessing
• Data Cleaning- remove single quotation and double quo- tations such as b’ and b”. In our exploration of data,the full text column always starts with the string b’ or b” . We affirm that the string does contain b’ or b”. And, we want to focus on only the content text message.
• Remove punctuation -Define a method called clean text to remove unwanted and some uncorrelated punctua- tion. In the method, we create a translation table using str.maketrans() method that returns a translation table with a 1-to-1 mapping of a Unicode ordinal to its trans- lation/replacement.To be specific, to create a translation of each punctuation that wants to None. Then we apply the method to both training data and testing data.
• Remove stop words - Import stop-words from nltk. Im- plement list comprehension to check if each word is not a stop-word, not ’http’ , not a digit and not include any of unwanted stop words, then return to a lower case, and strip quotation marks string.
• Reinforce hashtag content - Hashtag usually is the topic of the tweets and expresses what each tweet will be discuss. Emphasize hashtag content to make sure the model values the importance of tweets. We took whatever in the hashtag and restated at the end of each full text.
B. Feature Engineering
When predicting each tweet belongs to either democrat or republican, the most important metric is the full text column because it contains the full text in each tweet, it has almost all the information we need. Also, the hashtags column is helpful because it emphasizes some words in the full text, which means it adds more weight to specific words when we are using all those features (or words) to do prediction. However, we did not use the favorite count and retweet count columns. we thought they may not have a really big impact on the prediction because favorite and retweet may depend on each person’s preference.
C. Analysis and model
• Tf–idf Vectorizer - First, we used the tf–idf vectorizer to convert the full text column in the data-set to a matrix of TF–IDF features. TF–IDF will tokenize documents, learn the vocabulary and inverse document frequency weightings, and allow us to encode new documents. It is better than count vectorizer because it focuses on the frequency of words present in our data-set and provides the importance of the words instead of simply converting a set of strings into a frequency representation. We selected ngram range to be (1,1). After several attempts, we found out that the maximum features set to 400000 can be the best number to evaluate the accuracy.
• Logistic regression - Since the objective was to predict the party for each tweet, we constructed the logistic regres- sion model. It is a supervised machine learning technique for classification problems. It trains on a labeled data-set along with the answer key, which is the party in this data- set, to train and evaluate its accuracy. This model can help us to learn and approximate a mapping function f(Xi) = Y from input variables x1,x2,xn to the output variable (Y). In this data-set, we fit our X train (full text and hashtags) and y train (party) set to get the prediction of the X test data-set. Also, it seems to be more efficient than other models because its run time is much less than others such as SVM and random forest classifier. We set the cv as 5 and the maximum iteration as 50, as well as used the default lbfgs solver. We used the default solver since it has the highest accuracy compared to other solvers.


CLASSIFICATION RESULT:
A. Discussion
We are not sure whether the model is over-fitting or not because we trained with too many detailed tweets and given a lot of information in training data which would increase some bias. And since the model is very simple such that it can not get more detailed information. So maybe be it is not over-fitting.
Our highest accuracy score is 0.89247. At the beginning, our score was about 0.86. But we selected different numbers of features when applied the tfidfVectorizer and changed the maximum iteration when fit in the logistic regression, then the accuracy became higher
B. Self-Reflection
We did not use more complex models or algorithms, such as neural networks. Besides that, we did not use optimizers, such as gradient descent. There are several types of gradient descent approaches we can try to use, for example, stochastic gradient descent, mini-batch gradient descent, AdaGrad, RMS- Prop, and so on. In addition, our feature selections may not be perfect. It may be improved if we do better feature engineering.
C. Acknowledgment
We would like to thank Dr.Caliskan. In our coursework he taught us that used the nltk package and logistic regression to predict text data. Thanks for being a predictor of thought. We’d also like to thank the T.A’s for their time and guidance for this competition.
