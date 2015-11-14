# Word2Vec

Google's Word2Vec is a deep-learning inspired method that focuses on the meaning of words. Word2Vec attempts to understand meaning and semantic relationships among words. It works in a way that is similar to deep approaches, such as recurrent neural nets or deep neural nets, but is computationally more efficient.
Here are the steps that I have taken

**a.	Preprocessing**

The purpose of this step is to clean the data of anything that is considered irrelevant. Mostly these are stopwords, special symbols and html tags. However, it might also be prudent to keep some of the special symbols and stopwords. The reasons for this will become apparent later.

`cleanup.py contains the script and functions to perfoem data cleanup`
