# Churn Prediction

Code: Churn_Prediction.py

This is a project to predict the churn in an organization. 

Key Points:
1. Clean data has been used as input
2. Models used: Logistic Regression, Random Forest and SVM
3. GridSearchCV has been used for hyperparameter optimization
4. Pipeline has been used for ease of model execution and finding the best performing model based on accurracy.

Disclaimer: Pipeline related code has been inspired by Matthew Mayo's blog on KDNuggests.  

# Model Optimization with Hyperparameter Tuning

Further reading:  
https://machinelearningmastery.com/configure-gradient-boosting-algorithm/  
https://shankarmsy.github.io/stories/gbrt-sklearn.html  


## GBM ##
Further reading:  
https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/  
https://nycdatascience.com/blog/meetup/featured-talk-1-kaggle-data-scientist-owen-zhang/  
https://shankarmsy.github.io/stories/gbrt-sklearn.html  
https://effectiveml.com/using-grid-search-to-optimise-catboost-parameters.html  
https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html  


# Catboost #

Code: Catboost.py  

When the dataset contains many categorical features, catboost seems to be working pretty well among other gradient boosting based algorithms. With catboost, there is no need of explicit one-hot encoding.  

Disclaimer: Some part of the code has been reused from other posts.  

## Chatbot ##

Code: Chatbot.py

There are numerous use cases where a simple conversational chatbot can fetch the data from a table based on the input criteria mentioned in the user input. NLTK can be used to clean and normalize both the tabular data and user input for attribute and value selection. Column names can be segregated as entities and intents for the bot to operate. Moreover the type of the operation can read as intent from the user input.  

Further reading: https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e  



## Insights ##

Code: Sophisticated Excel Reporting in Python.py, Excel Reporting in Python.py  

While working on data science project there would be circumstances, team needs to provide repeatitive reports as insights for business to compare how the data science projects are helping them. It would be time consuming task to create such reports manually everytime. We can automate the task in Python that would create fancy excel report for business to look into.  

