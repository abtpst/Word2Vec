import pickle
import time
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import utilities.preProc as preProc
import utilities.classifierFuncs as cfun

def main():
    
    modelName = "../../classifier/Word2VectforNLPTraining"
    model = Word2Vec.load(open(modelName,"rb"))

    # model.init_sims(replace=True)

    wordVectors = model.syn0
    # print(wordVectors[0])
    num_clusters = int(wordVectors.shape[0] / 5)
    # print("number of clusters: {}".format(num_clusters))
    # input("Press enter to continue:")
    print("Clustering...")
    startTime = time.time()
    clusterIndex = cfun.kmeans(num_clusters, wordVectors)
    endTime = time.time()

    print("Time taken for clustering: {} seconds".format(endTime - startTime))

    clusterf = open("../../classifier/doc2vec/clusterIndex.pickle","wb") 
    
    pickle.dump(clusterIndex,clusterf)
    
    # create a word/index dictionary, mapping each vocabulary word to a cluster number
    # zip(): make an iterator that aggregates elements from each of the iterables
    index_word_map = dict(zip(model.index2word, clusterIndex))

    train = pd.read_csv("../../data/labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("../../data/testData.tsv",
                   header=0, delimiter="\t", quoting=3)

    trainingDataFV = np.zeros((train["review"].size, num_clusters), dtype=np.float)
    testDataFV = np.zeros((test["review"].size, num_clusters), dtype=np.float)

    # We dont really need to clean the data as the junk terms will be ignored anyway. This is due to the fact that we did not consider these while creating the model
    # and hence they will not feature in the model's vocabulary. Still this step will expedite the classification and feature vector creation.
    print("Processing training data...")
    counter = 0
    cleanedTrainingData = preProc.clean_data(train)
    for review in cleanedTrainingData:
        trainingDataFV[counter] = cfun.create_bag_of_centroids(review,num_clusters,index_word_map)
        counter += 1

    print("Processing test data...")
    counter = 0
    cleaned_test_data = preProc.clean_data(test)
    for review in cleaned_test_data:
        testDataFV[counter] = cfun.create_bag_of_centroids(review,num_clusters,index_word_map)
        counter += 1

    n_estimators = 100
    result = cfun.rfClassifer(n_estimators, trainingDataFV, train["sentiment"],testDataFV)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("Word2Vec_Clustering.csv", index=False, quoting=3)

if __name__ == '__main__':
    main()
