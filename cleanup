import pandas as pd
import json
import utilities.preProc as preProc 
import nltk.data
from gensim.models import word2vec

if __name__ == '__main__':
    
    # Read labeled trining data
    
    train = pd.read_csv("../../data/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
    
    # Read unlabeled trining data
    
    unlabeled_train = pd.read_csv("../../data/unlabeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
    
    # Load a tokenizer from nltk
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Initialize list for holding cleaned up sentences
    
    bagOfsentences = []
    
    # Parse labeled sentences and append to bagOfsentences
    
    print("Parsing sentences from labeled training set")
    for review in train["review"]:
        bagOfsentences.append(preProc.review_to_sentences(review, tokenizer, False, True, False))
    
    # Parse unlabeled sentences and append to bagOfsentences
    
    print("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        bagOfsentences.append(preProc.review_to_sentences(review, tokenizer, False, True, False))
    
    # Save bagOfsentences
    
    json.dump(bagOfsentences,open("../../classifier/bagOfsentences.json", "a"))

# Here is the function review_to_sentences 

def review_to_sentences(review, tokenizer, sentiment="",removeStopwords=False, removeNumbers=False, removeSmileys=False):
    """
    This function splits a review into parsed sentences
    :param review:
    :param tokenizer:
    :param removeStopwords:
    :return: sentences, list of lists
    """
    # review.strip()remove the white spaces in the review
    # use tokenizer to separate review to sentences
    
    rawSentences = tokenizer.tokenize(review.strip())

    cleanedReview = []
    for sentence in rawSentences:
        if len(sentence) > 0:
            cleanedReview += review_to_words(sentence, removeStopwords, removeNumbers, removeSmileys)

    if(sentiment != ""):
        cleanedReview.append(sentiment)
              
    return cleanedReview

#The function review_to_words

def review_to_words(rawReview, removeStopwords=False, removeNumbers=False, removeSmileys=False):
    
    # use BeautifulSoup library to remove the HTML/XML tags (e.g., <br />)
    reviewText = BeautifulSoup(rawReview).get_text()

    # Emotional symbols may affect the meaning of the review
    smileys = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^)
                :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    smiley_pattern = "|".join(map(re.escape, smileys))

    # [^] matches a single character that is not contained within the brackets
    # re.sub() replaces the pattern by the desired character/string
    
    if removeNumbers and removeSmileys:
        # any character that is not in a to z and A to Z (non text)
        reviewText = re.sub("[^a-zA-Z]", " ", reviewText)
    elif removeSmileys:
         # numbers are also included
        reviewText = re.sub("[^a-zA-Z0-9]", " ", reviewText)
    elif removeNumbers:
        reviewText = re.sub("[^a-zA-Z" + smiley_pattern + "]", " ", reviewText)
    else:
        reviewText = re.sub("[^a-zA-Z0-9" + smiley_pattern + "]", " ", reviewText)


    # split in to a list of words
    words = reviewText.lower().split()

    if removeStopwords:
        # create a set of all stop words
        stops = set(stopwords.words("english"))
        # remove stop words from the list
        words = [w for w in words if w not in stops]

    # for bag of words, return a string that is the concatenation of all the meaningful words
    # for word2Vector, return list of words
    # return " ".join(words)
               
    return words
