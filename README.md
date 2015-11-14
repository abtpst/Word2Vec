# Word2Vec

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


