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
import pandas as pd
import logging
import numpy as np
import sys
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk import ngrams
from nltk.corpus import wordnet
from nltk import RegexpParser
from textblob import TextBlob
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import urllib.request
import zipfile
import os
from os.path import expanduser
import platform

from text import TextCleaning as text_cleaner
from text.Embedding import extractFeatureUsingPreTrainedModel

logEnabled = False
spacy_nlp = None
        
def ExtractFeatureCountVectors(ngram_range=(1, 1),
                               max_df=1.0,
                               min_df=1,
                               max_features=None,
                               binary=False):
    vectorizer = CountVectorizer(ngram_range = ngram_range, max_df = max_df, \
                                 min_df = min_df, max_features = max_features, binary = binary)
    return vectorizer            

def ExtractFeatureTfIdfVectors(ngram_range=(1, 1), 
                               max_df=1.0, 
                               min_df=1, 
                               max_features=None,
                               binary=False,
                               norm='l2',
                               use_idf=True,
                               smooth_idf=True,
                               sublinear_tf=False):
    vectorizer = TfidfVectorizer(ngram_range = ngram_range, max_df = max_df, \
                                 min_df = min_df, max_features = max_features, \
                                 binary = binary, norm = norm, use_idf = use_idf, \
                                 smooth_idf = smooth_idf, sublinear_tf = sublinear_tf)
    return vectorizer            

def GetPOSTags( inputText, getPOSTags_Lib='nltk'):
    global spacy_nlp
    tokens_postag_list = []
    
    if (inputText == ""):
        __Log("debug", "{} function: Input text is not provided".format(sys._getframe().f_code.co_name))
    else:
        if getPOSTags_Lib == 'spacy':
            if spacy_nlp == None:
                spacy_nlp = spacy.load('en_core_web_sm')                        
            doc = spacy_nlp(inputText)
            for token in doc:
                tokens_postag_list.append((token.text, token.tag_))
        elif getPOSTags_Lib == 'textblob':
            doc = TextBlob(inputText)
            tokens_postag_list = doc.tags
        else:
            tokensList = WordTokenize(inputText)
            tokens_postag_list = pos_tag(tokensList)
            
    return tokens_postag_list
        
def GetNGrams( inputText, ngramRange=(1,1)):
    ngramslist = []
    for n in range(ngramRange[0],ngramRange[1]+1):
        nwordgrams = ngrams(inputText.split(), n)
        ngramslist.extend([' '.join(grams) for grams in nwordgrams])
    return ngramslist      
        
def NamedEntityRecognition( inputText):
    global spacy_nlp
    neResultList = []
    if (inputText == ""):
        __Log("debug", "{} function: Input text is not provided".format(sys._getframe().f_code.co_name))
    else:
        if spacy_nlp == None:
            spacy_nlp = spacy.load('en_core_web_sm') 
        doc = spacy_nlp(inputText)
        neResultList = [(X.text, X.label_) for X in doc.ents]
        
    return neResultList
        
def KeywordsExtraction( inputText, ratio=0.2, words = None, scores=False, pos_filter=('NN', 'JJ'), lemmatize=False):
    keywordsList = []
    if (inputText == ""):
        __Log("debug", "{} function: Input text is not provided".format(sys._getframe().f_code.co_name))
    else:
        keywordsList = keywords(inputText, ratio = ratio, words = words, split=True, scores=scores, 
                            pos_filter=pos_filter, lemmatize=lemmatize)
    return keywordsList

def __get_nodes(parent): 
    nounList = []
    verbList = []
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == "NP":
                subList = []
                for item in node.leaves():
                    subList.append(item[0])
                nounList.append((" ".join(subList)))
            elif node.label() == "VP":
                subList = []
                for item in node.leaves():
                    subList.append(item[0])
                verbList.append((" ".join(subList)))
                #verbList.append(node.leaves()[0][0])
            __get_nodes(node)
    result = {'NP': nounList, 'VP': verbList}
    return result

def ShallowParsing( inputText, lib='spacy'):
    tags = GetPOSTags(inputText, getPOSTags_Lib=lib)
    
    chunk_regex = r"""
        NBAR:
            {<DT>?<NN.*|JJ.*>*<NN.*>+}  # Nouns and Adjectives, terminated with Nouns
        VBAR:
            {<RB.?>*<VB.?>*<TO>?<JJ>*<VB.?>+<VB>?} # Verbs and Verb Phrases
            
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        VP: 
            {<VBAR>}
            {<VBAR><IN><VBAR>}  # Above, connected with in/of/etc...
    """
    rp = RegexpParser(chunk_regex)
    t = rp.parse(tags)
    return __get_nodes(t)

def SyntacticAndEntityParsing(inputCorpus, 
                              featuresList=['POSTags','NGrams','NamedEntityRecognition','KeywordsExtraction','ShallowParsing'],
                              posTagsLib='nltk', 
                              ngramRange=(1,1), 
                              ke_ratio=0.2, 
                              ke_words = None, 
                              ke_scores=False, 
                              ke_pos_filter=('NN', 'JJ'), 
                              ke_lemmatize=False):
    columnsList = ['Input']
    columnsList.extend(featuresList)
    df = pd.DataFrame(columns=columnsList)
    df['Input'] = inputCorpus
    for feature in featuresList:
        if feature == 'POSTags':
            df[feature] = inputCorpus.apply(lambda x: GetPOSTags(x, posTagsLib))
        if feature == 'NGrams':
            df[feature] = inputCorpus.apply(lambda x: GetNGrams(x, ngramRange))
        if feature == 'NamedEntityRecognition':
            df[feature] = inputCorpus.apply(lambda x: NamedEntityRecognition(x))
        if feature == 'KeywordsExtraction':
            df[feature] = inputCorpus.apply(lambda x: KeywordsExtraction(x, 
                                              ratio=ke_ratio, words=ke_words,
                                              scores=ke_scores, pos_filter=ke_pos_filter,
                                              lemmatize=ke_lemmatize))
        if feature == 'ShallowParsing':
            df[feature] = inputCorpus.apply(lambda x: ShallowParsing(x, lib=posTagsLib))
    return df
    
def __Log( logType="info", text=None):
        if logType.lower() == "exception":
            logging.exception( text)
        elif logEnabled:
            if logType.lower() == "info":
                logging.info( text)
            elif logType.lower() == "debug":
                logging.debug( text)

def SentenceTokenize( inputText):
    return text_cleaner.WordTokenize(inputText)

def WordTokenize( inputText, tokenizationLib = 'nltk'):
    return text_cleaner.WordTokenize(inputText, tokenizationLib)

def Lemmatize( inputTokensList, lemmatizationLib = 'nltk'):
    return text_cleaner.Lemmatize(inputTokensList, lemmatizationLib)

def Stemmize( inputTokensList):
    return text_cleaner.Stemmize(inputTokensList)
    
def ToLowercase( inputText):
    resultText = ""
    if inputText is not None and inputText != "":
        resultText = inputText.lower()
    return resultText

def ToUppercase( inputText):
    resultText = ""
    if inputText is not None and inputText != '':   
        resultText = inputText.upper()
    return resultText
    
def RemoveNoise(
                inputText, 
                removeNoise_fHtmlDecode = True, 
                removeNoise_fRemoveHyperLinks = True,
                removeNoise_fRemoveMentions = True, 
                removeNoise_fRemoveHashtags = True, 
                removeNoise_RemoveOrReplaceEmoji = 'remove',
                removeNoise_fUnicodeToAscii = True,
                removeNoise_fRemoveNonAscii = True):
    return text_cleaner.RemoveNoise(inputText, removeNoise_fHtmlDecode, removeNoise_fRemoveHyperLinks, removeNoise_fRemoveMentions,
    removeNoise_fRemoveHashtags, removeNoise_RemoveOrReplaceEmoji, removeNoise_fUnicodeToAscii, removeNoise_fRemoveNonAscii)
    
def RemoveStopwords( inputTokensList, stopwordsRemovalLib='nltk', stopwordsList = None, extend_or_replace='extend'):
    return text_cleaner.RemoveStopwords(inputTokensList, stopwordsRemovalLib, stopwordsList, extend_or_replace)
    
def RemoveNumericTokens( inputText, removeNumeric_fIncludeSpecialCharacters=True):
    return text_cleaner.RemoveNumericTokens(inputText, removeNumeric_fIncludeSpecialCharacters)
    
def RemovePunctuation( inputText, fRemovePuncWithinTokens=False):
    return text_cleaner.RemovePunctuation(inputText, fRemovePuncWithinTokens)
    
def CorrectSpelling( inputTokensList):
    return text_cleaner.CorrectSpelling(inputTokensList)
    
def ReplaceAcronym( inputTokensList, acrDict=None):
    return text_cleaner.ReplaceAcronym(inputTokensList, acrDict)
    
def ExpandContractions( inputText, expandContractions_googleNewsWordVectorPath=None):
    return text_cleaner.ExpandContractions(inputText, expandContractions_googleNewsWordVectorPath)
        
def get_pretrained_model_path():
    try:
        from appbe.dataPath import DATA_DIR
        modelsPath = Path(DATA_DIR)/'PreTrainedModels'/'TextProcessing'
    except:
        modelsPath = Path('aion')/'PreTrainedModels'/'TextProcessing'
    if not modelsPath.exists():
        modelsPath.mkdir(parents=True, exist_ok=True)
    return modelsPath
    
def checkAndDownloadPretrainedModel(preTrainedModel, embedding_size=300):
    
    models = {'glove':{50:'glove.6B.50d.w2vformat.txt',100:'glove.6B.100d.w2vformat.txt',200:'glove.6B.200d.w2vformat.txt',300:'glove.6B.300d.w2vformat.txt'}, 'fasttext':{300:'wiki-news-300d-1M.vec'}}
    supported_models = [x for y in models.values() for x in y.values()]
    embedding_sizes = {x:y.keys() for x,y in models.items()}
    preTrainedModel = preTrainedModel.lower()
    if preTrainedModel not in models.keys():
        raise ValueError(f'model not supported: {preTrainedModel}')
    if embedding_size not in embedding_sizes[preTrainedModel]:
        raise ValueError(f"Embedding size '{embedding_size}' not supported for {preTrainedModel}")
    selected_model = models[preTrainedModel][embedding_size]
    modelsPath = get_pretrained_model_path()
    p = modelsPath.glob('**/*')
    modelsDownloaded = [x.name for x in p if x.name in supported_models]
    if selected_model not in modelsDownloaded:
        if preTrainedModel == "glove":
            try:
                local_file_path =  modelsPath/f"glove.6B.{embedding_size}d.w2vformat.txt"
                file_test, header_test = urllib.request.urlretrieve(f'https://aion-pretrained-models.s3.ap-south-1.amazonaws.com/text/glove.6B.{embedding_size}d.w2vformat.txt', local_file_path)
            except Exception as e:
                raise ValueError("Error: unable to download glove pretrained model, please try again or download it manually and placed it at {}. ".format(modelsPath)+str(e))

        elif preTrainedModel == "fasttext":
            try:
                local_file_path =  modelsPath/"wiki-news-300d-1M.vec.zip"
                url = 'https://aion-pretrained-models.s3.ap-south-1.amazonaws.com/text/wiki-news-300d-1M.vec.zip'
                file_test, header_test = urllib.request.urlretrieve(url, local_file_path)
                with zipfile.ZipFile(local_file_path) as zip_ref:
                    zip_ref.extractall(modelsPath)
                Path(local_file_path).unlink()
            except Exception as e:
                raise ValueError("Error: unable to download fastText pretrained model, please try again or download it manually and placed it at {}. ".format(location)+str(e))
    return modelsPath/selected_model
    
def load_pretrained(path):
    embeddings = {}
    word = ''
    with open(path, 'r', encoding="utf8") as f:
        header = f.readline()
        header = header.split(' ')
        vocab_size = int(header[0])
        embed_size = int(header[1])
        for i in range(vocab_size):
            data = f.readline().strip().split(' ')
            word = data[0]
            embeddings[word] = [float(x) for x in data[1:]]
    return embeddings

class TextProcessing(BaseEstimator, TransformerMixin):
    
    def __init__(self,
            functionSequence = ['RemoveNoise','ExpandContractions','Normalize','ReplaceAcronym',
                'CorrectSpelling','RemoveStopwords','RemovePunctuation','RemoveNumericTokens'],
            fRemoveNoise = True,
            fExpandContractions = False,
            fNormalize = True,
            fReplaceAcronym = False,
            fCorrectSpelling = False,
            fRemoveStopwords = True,
            fRemovePunctuation = True,
            fRemoveNumericTokens = True,
            removeNoise_fHtmlDecode = True, 
            removeNoise_fRemoveHyperLinks = True,
            removeNoise_fRemoveMentions = True, 
            removeNoise_fRemoveHashtags = True,
            removeNoise_RemoveOrReplaceEmoji = 'remove',
            removeNoise_fUnicodeToAscii = True,
            removeNoise_fRemoveNonAscii = True,
            tokenizationLib='nltk',
            normalizationMethod = 'Lemmatization',
            lemmatizationLib = 'nltk',
            acronymDict = None,
            stopwordsRemovalLib = 'nltk',
            stopwordsList = None,
            extend_or_replace_stopwordslist = 'extend',
            removeNumeric_fIncludeSpecialCharacters = True,
            fRemovePuncWithinTokens = False,
            data_path = None
):
        global logEnabled
        #logEnabled = EnableLogging
        self.functionSequence = functionSequence
        self.fRemoveNoise = fRemoveNoise
        self.fExpandContractions = fExpandContractions
        self.fNormalize = fNormalize
        self.fReplaceAcronym = fReplaceAcronym
        self.fCorrectSpelling = fCorrectSpelling
        self.fRemoveStopwords = fRemoveStopwords
        self.fRemovePunctuation = fRemovePunctuation
        self.fRemoveNumericTokens = fRemoveNumericTokens
        self.removeNoise_fHtmlDecode = removeNoise_fHtmlDecode
        self.removeNoise_fRemoveHyperLinks = removeNoise_fRemoveHyperLinks
        self.removeNoise_fRemoveMentions = removeNoise_fRemoveMentions 
        self.removeNoise_fRemoveHashtags = removeNoise_fRemoveHashtags
        self.removeNoise_RemoveOrReplaceEmoji = removeNoise_RemoveOrReplaceEmoji
        self.removeNoise_fUnicodeToAscii = removeNoise_fUnicodeToAscii
        self.removeNoise_fRemoveNonAscii = removeNoise_fRemoveNonAscii
        self.tokenizationLib = tokenizationLib
        self.normalizationMethod = normalizationMethod
        self.lemmatizationLib = lemmatizationLib
        self.acronymDict = acronymDict
        self.stopwordsRemovalLib = stopwordsRemovalLib
        self.stopwordsList = stopwordsList
        self.extend_or_replace_stopwordslist = extend_or_replace_stopwordslist
        self.removeNumeric_fIncludeSpecialCharacters = removeNumeric_fIncludeSpecialCharacters
        self.fRemovePuncWithinTokens = fRemovePuncWithinTokens
        self.data_path = data_path
        self.fit_and_transformed_ = False

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        x = map(lambda inputText: text_cleaner.cleanText(inputText, functionSequence = self.functionSequence, fRemoveNoise = self.fRemoveNoise, fExpandContractions = self.fExpandContractions, fNormalize = self.fNormalize, fReplaceAcronym = self.fReplaceAcronym, fCorrectSpelling = self.fCorrectSpelling, fRemoveStopwords = self.fRemoveStopwords, fRemovePunctuation = self.fRemovePunctuation, fRemoveNumericTokens = self.fRemoveNumericTokens, removeNoise_fHtmlDecode = self.removeNoise_fHtmlDecode, removeNoise_fRemoveHyperLinks = self.removeNoise_fRemoveHyperLinks, removeNoise_fRemoveMentions = self.removeNoise_fRemoveMentions , removeNoise_fRemoveHashtags = self.removeNoise_fRemoveHashtags, removeNoise_RemoveOrReplaceEmoji = self.removeNoise_RemoveOrReplaceEmoji, removeNoise_fUnicodeToAscii = self.removeNoise_fUnicodeToAscii, removeNoise_fRemoveNonAscii = self.removeNoise_fRemoveNonAscii, tokenizationLib = self.tokenizationLib, normalizationMethod = self.normalizationMethod, lemmatizationLib = self.lemmatizationLib, acronymDict = self.acronymDict, stopwordsRemovalLib = self.stopwordsRemovalLib, stopwordsList = self.stopwordsList, extend_or_replace_stopwordslist = self.extend_or_replace_stopwordslist, removeNumeric_fIncludeSpecialCharacters = self.removeNumeric_fIncludeSpecialCharacters, fRemovePuncWithinTokens = self.fRemovePuncWithinTokens), x)
        x = pd.Series(list(x))
        if hasattr(self, 'fit_and_transformed_') and not self.fit_and_transformed_:
            self.fit_and_transformed_ = True
            if self.data_path and Path(self.data_path).exists():
                x.to_csv(Path(self.data_path)/'text_cleaned.csv', index=False)
        return x

    def get_feature_names_out(self):
        return ['tokenize']

class wordEmbedding(BaseEstimator, TransformerMixin):

    def __init__(self, preTrainedModel, embeddingSize=300,external_model=None,external_model_type='binary'):
        self.number_of_features = 0
        self.embeddingSize = embeddingSize
        self.preTrainedModel = preTrainedModel.lower()
        self.external_model=external_model
        self.external_model_type = external_model_type
        if self.preTrainedModel == "glove":
            self.preTrainedModelpath = f'glove.6B.{self.embeddingSize}d.w2vformat.txt'
            self.binary = False
        elif self.preTrainedModel == "fasttext":
            self.preTrainedModelpath = 'wiki-news-300d-1M.vec'
            self.binary = False
        else:
            raise ValueError(f'Model ({self.preTrainedModel}) not supported')
        
    def fit(self, x, y=None):
        return self
        
    def transform(self, x):
        if ((isinstance(self.external_model, pd.DataFrame) and not self.external_model.empty) or (not isinstance(self.external_model, pd.DataFrame) and self.external_model)):
            if self.preTrainedModel == "fasttext" and self.external_model_type == 'binary':
                print('Transforming using external binary')
                extracted = np.vstack([self.external_model.get_sentence_vector( sentense) for sentense in x])
            else:
                print('Transforming using external vector')
                extracted = extractFeatureUsingPreTrainedModel(x, pretrainedModelPath=None, loaded_model=self.external_model, embed_size=300)
        else:
            print('Transforming using Vector')
            models_path = checkAndDownloadPretrainedModel(self.preTrainedModel, self.embeddingSize)
            extracted = extractFeatureUsingPreTrainedModel(x, models_path)

        self.number_of_features = extracted.shape[1]
        return extracted
        
    def get_feature_names_out(self):
        return [str(x) for x in range(self.number_of_features)]

    def get_feature_names(self):
        return self.get_feature_names_out()

def getProcessedPOSTaggedData(pos_tagged_data):
    def get_wordnet_post(tag):
        if tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def process_pos_tagged_data(text):
        processed_text = [f"{t[0]}_{get_wordnet_post(t[1])}" for t in text]
        processed_text = " ".join(processed_text)
        return processed_text

    processed_pos_tagged_data = pos_tagged_data.apply(process_pos_tagged_data)
    return processed_pos_tagged_data
        
        
class PosTagging(BaseEstimator, TransformerMixin):

    def __init__(self, posTagsLib, data_path):
        self.posTagsLib = posTagsLib
        self.fit_and_transformed_ = False
        self.data_path = data_path
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        parsing_output = SyntacticAndEntityParsing(x, featuresList=['POSTags'], posTagsLib=self.posTagsLib)
        output = getProcessedPOSTaggedData(parsing_output['POSTags'])
        if not self.fit_and_transformed_:
            self.fit_and_transformed_ = True
            if self.data_path and Path(self.data_path).exists():
                output.to_csv(Path(self.data_path)/'pos_tagged.csv', index=False)
        return output
