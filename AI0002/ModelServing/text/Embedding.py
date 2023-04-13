'''
*
* =============================================================================
* COPYRIGHT NOTICE
* =============================================================================
*  @ Copyright HCL Technologies Ltd. 2021, 2022
* Proprietary and confidential. All information contained herein is, and
* remains the property of HCL Technologies Limited. Copying or reproducing the
* contents of this file, via any medium is strictly prohibited unless prior
* written permission is obtained from HCL Technologies Limited.
*
'''
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

# Private function
def unitvec(vec):
    return vec / np.linalg.norm(vec)
    
def __word_average(vectors, sent, vector_size):
    """
    Compute average word vector for a single doc/sentence.
    """
    try:
        mean = []
        words = [word for word in sent if word in vectors.index]
        if words:
            mean = vectors.loc[words].to_numpy()

        if len(mean):
            m = np.array(mean)
            m = m.mean(axis=0)
            mean = unitvec(m)
            return mean
        return np.zeros(vector_size)
    except:
        raise

# Private function
def __word_average_list(vectors, docs, embed_size):
    """
    Compute average word vector for multiple docs, where docs had been tokenized.
    """
    try:
        return np.vstack([__word_average(vectors, sent, embed_size) for sent in docs])
    except:
        raise

def load_pretrained(path):
    df = pd.read_csv(path, index_col=0,sep=' ',quotechar = ' ' , header=None, skiprows=1)
    return len(df.columns), df
    
def extractFeatureUsingPreTrainedModel(inputCorpus, pretrainedModelPath=None, loaded_model=False, embed_size=300):
    """
    Extract feature vector from input Corpus using pretrained Vector model(word2vec,fasttext, glove(converted to word2vec format)
    """
    try:
        if inputCorpus is None:           
            return None
        else:
            if not pretrainedModelPath and ((isinstance(loaded_model, pd.DataFrame) and loaded_model.empty) or (not isinstance(loaded_model, pd.DataFrame) and not loaded_model)):
                inputCorpusWordVectors = None
            else:
                if (isinstance(loaded_model, pd.DataFrame) and not loaded_model.empty) or loaded_model:
                    pretrainedModel = loaded_model
                    embed_size = embed_size
                else:
                    embed_size, pretrainedModel = load_pretrained(pretrainedModelPath)
                   
                if (isinstance(pretrainedModel, pd.DataFrame) and not pretrainedModel.empty) or pretrainedModel:
                    input_docs_tokens_list = [word_tokenize(inputDoc) for inputDoc in inputCorpus]
                    inputCorpusWordVectors = __word_average_list(pretrainedModel, input_docs_tokens_list,embed_size)
                else:
                    inputCorpusWordVectors = None
            return inputCorpusWordVectors
    except:
        raise  
