# Word2Vec

I have prepared the code base to tackle the problem of sentiment analysis. I believe that the methods that I explore here can be modified to perform document tagging. Basically, both problems involve looking at a piece of text and categorizing it. We will look at binary sentiment analysis i.e. good or bad. 

The problem set is taken from

https://www.kaggle.com/c/word2vec-nlp-tutorial

The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. In addition, there are another 50,000 IMDB reviews provided without any rating labels.

Google's Word2Vec is a deep-learning inspired method that focuses on the meaning of words. Word2Vec attempts to understand meaning and semantic relationships among words. It works in a way that is similar to deep approaches, such as recurrent neural nets or deep neural nets, but is computationally more efficient.
Here are the steps that I have taken

**a.	Preprocessing**

The purpose of this step is to clean the data of anything that is considered irrelevant. Mostly these are stopwords, special symbols and html tags. However, it might also be prudent to keep some of the special symbols and stopwords. The reasons for this will become apparent later.

`dataCleanup.py contains the script and functions to perform data cleanup`

Note that I am indicating that I do not want to remove numbers and smileys. This makes sense as things like 9/10 or  can give clues about the sentiment of a review. Turns out, Word2Vec is smart enough to pick up these clues as we shall soon see.

Here is a summary of what we achieved in this step

i.	Break up the reviews into individual sentences

ii. From individual sentences, take out all text that is considered irrelevant.

iii. Finally return a list of relevant words/tokens representing each review.

Now we are ready to train our model. Also, notice that our training set consists of both labeled and unlabeled data. Another advantage of Word2Vec in this context is that it will be able to form meaningful relationships between words even without explicit knowledge of the tags. More on this later.

**b.	Training**

`trainModel.py trains and stores the word2vec model`

Please go through the well documented script to understand how the word2vec model is built.

Main features of this model are

i.	We want all word vectors to have a dimension of 500. In other words, each word vector has 500 features.

ii.	We are ignoring any word that appears less than 40 times in our corpus.

iii.	For each review or text we analyze, we are looking at phrases of 10 words at most.

Now let’s review what all of this gives us

**c.	Testing and analyzing the model**

Word2Vec’s gensim implementation has some really nice functions that let us ‘look’ into the model. Here is a python script for exploring the model

`testModel.py contains the script to uncover the model`

First, let’s look at what is the data type of the model

`print(type(model.syn0))`

`<class 'numpy.ndarray'>`

Now, let’s see how many word vectors has our model created. This can be viewed by

`print(type(model.syn0.shape))`

`(16982, 500)`

So it’s a numpy array with 500 columns i.e. one for each feature as we defined in step b. It also has 16982 rows. Meaning our model has created 16982 word vectors form the entire dataset that we used to train it. Let’s see if these are enough to derive useful information. Here is the result of 

`print(model.most_similar("garbage"))`

`[('crap', 0.6813626289367676), ('trash', 0.6237289905548096), ('junk', 0.6178079843521118), ('rubbish', 0.5900264382362366), ('pile', 0.5394319891929626), ('utter', 0.53409343957901), ('dreck', 0.5119433403015137), ('tripe', 0.5090062618255615), ('drivel', 0.5011706352233887), ('steaming', 0.48643001914024353)]`

pretty useful! Looks like our model was able to pick up ‘meaningful’ words after all and relate them together.

let’s look at another example

`print(model.most_similar(positive=['woman', 'boy'], negative=['man'], topn=10))`

here we want to find out the word vectors we get after adding the word vectors for boy and woman and subtracting the word vector for man. Here’s what we get

`most similar:
[('girl', 0.573335587978363), ('daughter', 0.3910765051841736), ('young', 0.36802443861961365), ('mother', 0.3579619526863098), ('teenage', 0.35339730978012085), ('baby', 0.34497135877609253), ('her', 0.3399488627910614), ('meets', 0.3394812345504761), ('pregnant', 0.3362891674041748), ('sister', 0.3335306644439697)]`

Pretty neat! It is important to note that we did not provide any semantic information to our model. Anything that the model derives is purely based on the spatial correlation of words/tokens in our data. As long as there is enough data, Word2Vec will be able to derive relationships that we can use. Finally, let’s find out if our earlier intuition of including smileys in our data set pays off

`print(model.most_similar(":-)"))`

`[(':d', 0.5370886325836182), ('ps:', 0.5244550108909607), ('nickelodeon', 0.5076525807380676), ('lotr', 0.5034156441688538), ('haha', 0.500064492225647), ('myself)', 0.49700990319252014), ('say:', 0.4925299286842346), ("(you'll", 0.4870578646659851), ('boobies', 0.48568087816238403), ('advice:', 0.4825150966644287)]`

Looks like we were right! We can get useful information from smileys. Note that not all smileys would be featured in the model. This will depend on the data. The more diverse data set we have, the better our model will be.

Finally, let’s test our model and see how it does on the test set.

**d.	Building the classifier**

Here we will build a classifier that will take a new piece of text and classify it as positive or negative. We will be creating a RandomForest classifier.  Also, we will be using K-Means clustering to create feature vectors for our training and test sets. Let’s break down this process 

i.	The classifier will need to look at training and test data to decide the sentiment tag.

ii.	Our model will aid the classifier in determining the terms that should be considered relevant. Remember that we said that our model created 16982 wordvectors. This is the vocabulary of the model. Note that the classifier and the model will use the same training data. So, if we have confidence in our model, with respect to our training set, then we can use our model as a reference to find out what is relevant and what is not.

iii.	So, we will look at all of the labeled training and test data. Note that, for the classifier, unlabeled training data is of no use. We will create feature vectors for each review in our labeled training set and our test set. While creating the feature vectors, we will ignore any terms that do not appear in the vocabulary of our model. This is the crucial step where our model comes into play.

iv.	Finally, we can feed the training set feature vectors to the classifier, along with the sentiment tags. Next we feed the test set feature vectors and let the classifier come up with the sentiment tags.

` randomForest.py the python script that does all of that`
`classifierFuncs.py is the helper script for clustering and other functions`

**e.	Evaluating the classifier**

This classifier gives me 84% accuracy on the test set. Not bad really. We can tweak our model’s parameters to see if we gain anything. 

