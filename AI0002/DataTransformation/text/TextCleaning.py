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

import re
import string
import sys
import demoji
#demoji.download_codes()
import nltk
import spacy
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from text_unidecode import unidecode
from textblob import TextBlob
from spellchecker import SpellChecker
from nltk import pos_tag
from nltk.tokenize import word_tokenize 
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from spacy.lang.en import English
from collections import defaultdict
import contractions


spacy_nlp = None

def WordTokenize( inputText, tokenizationLib = 'nltk'):
    tokenList = []
    if inputText is not None and inputText != "":
        tokenizationLib = tokenizationLib.lower()
        if tokenizationLib == 'nltk':
            tokenList = word_tokenize(inputText)
        elif tokenizationLib == 'textblob':
            tbObj = TextBlob(inputText)
            tokenList = tbObj.words
        elif tokenizationLib == 'spacy':
            nlp = English()
            nlpDoc = nlp(inputText)
            for token in nlpDoc:
                tokenList.append(token.text)
        elif tokenizationLib == 'keras':
            from tensorflow.keras.preprocessing.text import text_to_word_sequence
            tokenList = text_to_word_sequence(inputText)
        else:
            tokenList = word_tokenize(inputText)
        
    return tokenList
        
def SentenceTokenize( inputText):
    sentenceList = []
    if inputText is not None and inputText != "":
        sentenceList = sent_tokenize(inputText)
    return sentenceList

def Lemmatize(inputTokensList, lemmatizationLib = 'nltk'):
    global spacy_nlp
    lemmatized_list= []
    lemmatizationLib = lemmatizationLib.lower()
    if (inputTokensList is not None) and (len(inputTokensList)!=0):
        if (lemmatizationLib == 'textblob'):
            inputText = " ".join(inputTokensList)
            sent = TextBlob(inputText)
            tag_dict = {"J": 'a', 
                        "N": 'n', 
                        "V": 'v', 
                        "R": 'r'}
            words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags] 
            lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
        if (lemmatizationLib == 'spacy'):
            inputText = " ".join(inputTokensList)
            if spacy_nlp == None:
                spacy_nlp = spacy.load('en_core_web_sm')
            doc = spacy_nlp(inputText)
            
            for token in doc:
                if token.text != token.lemma_:
                    if token.lemma_ != "-PRON-":
                        lemmatized_list.append(token.lemma_)
                    else:
                        lemmatized_list.append(token.text)
                else:
                    lemmatized_list.append(token.text)
        else:
            tag_map = defaultdict(lambda : wordnet.NOUN)
            tag_map['J'] = wordnet.ADJ
            tag_map['V'] = wordnet.VERB
            tag_map['R'] = wordnet.ADV
        
            wnLemmatizer = WordNetLemmatizer()
            token_tags = pos_tag(inputTokensList)
            lemmatized_list = [wnLemmatizer.lemmatize(token, tag_map[tag[0]]) for token, tag in token_tags]

    return lemmatized_list
    
def Stemmize(inputTokensList):
    stemmedTokensList= []
    
    if (inputTokensList is not None) and (len(inputTokensList)!=0):
        porterStemmer  = PorterStemmer()
        stemmedTokensList = [porterStemmer.stem(token) for token in inputTokensList]
    
    return stemmedTokensList
        
def ToLowercase(inputText):
    resultText = ""
    if inputText is not None and inputText != "":
        resultText = inputText.lower()
    
    return resultText

def ToUppercase(inputText):
    resultText = ""
    if inputText is not None and inputText != '':   
        resultText = inputText.upper()
    
    return resultText
            
def RemoveNoise(inputText, 
                removeNoise_fHtmlDecode = True, 
                removeNoise_fRemoveHyperLinks = True,
                removeNoise_fRemoveMentions = True, 
                removeNoise_fRemoveHashtags = True, 
                removeNoise_RemoveOrReplaceEmoji = 'remove',
                removeNoise_fUnicodeToAscii = True,
                removeNoise_fRemoveNonAscii = True):
    if inputText is not None and inputText != "": 
        if removeNoise_fHtmlDecode == True:
            inputText = BeautifulSoup(inputText, "html.parser").text
        if removeNoise_fRemoveHyperLinks == True:
            inputText = re.sub(r'https?:\/\/\S*', '', inputText, flags=re.MULTILINE)            
        if removeNoise_fRemoveMentions == True:
            inputText = re.sub('[@]+\S+','', inputText)
        if removeNoise_fRemoveHashtags == True:
            inputText = re.sub('[#]+\S+','', inputText)
        if removeNoise_RemoveOrReplaceEmoji == 'remove':
            inputText = demoji.replace(inputText, "")
        elif removeNoise_RemoveOrReplaceEmoji == 'replace':
            inputText = demoji.replace_with_desc(inputText, " ")
        if removeNoise_fUnicodeToAscii == True:
            inputText = unidecode(inputText)
        if removeNoise_fRemoveNonAscii == True:
            inputText= re.sub(r'[^\x00-\x7F]+',' ', inputText)
        
        inputText = re.sub(r'\s+', ' ', inputText)
        inputText = inputText.strip()
    
    return inputText

def RemoveStopwords(inputTokensList, stopwordsRemovalLib='nltk', stopwordsList = None, extend_or_replace='extend'):
    resultTokensList = []
    if (inputTokensList is not None) and (len(inputTokensList)!=0):
        stopwordsRemovalLib= stopwordsRemovalLib.lower()
      
        if stopwordsRemovalLib == 'spacy':
            nlp = English()
            stopwordRemovalList = nlp.Defaults.stop_words
        else:
            stopwordRemovalList = set(stopwords.words('english'))
            
        if extend_or_replace == 'replace':
            if stopwordsList is not None:
                stopwordRemovalList = set(stopwordsList)
        else:
            if stopwordsList:
                stopwordRemovalList = stopwordRemovalList.union(set(stopwordsList))
            
        resultTokensList = [word for word in inputTokensList if word not in stopwordRemovalList] 
    
    return resultTokensList           
        
def RemoveNumericTokens(inputText, removeNumeric_fIncludeSpecialCharacters=True):
    resultText = ""
    if inputText is not None and inputText != "": 
        if removeNumeric_fIncludeSpecialCharacters == True:
            #Remove tokens having numbers and punctuations                    
            resultText = re.sub(r'\b\d+[^a-zA-Z]*\d*\b',' ', inputText)
        else:
            #Remove only numeric tokens
            resultText = re.sub(r'\b\d+\b','', inputText)
            
        # convert consecutive whitespaces to single space in the results
        resultText = re.sub(r'\s+', ' ', resultText)
    
    return resultText
    
def RemovePunctuation(inputText, fRemovePuncWithinTokens=False):
    resultText = ""
    if inputText is not None and len(inputText) != 0:
        if fRemovePuncWithinTokens == True:
            resultText = inputText.translate(str.maketrans("","", string.punctuation))
        else:
            punctuationList = list(string.punctuation)
            tokensList = WordTokenize(inputText)                
            resultTokensList = [word for word in tokensList if word not in punctuationList]               
            resultText = " ".join(resultTokensList)            
        
        resultText = re.sub(r'\s+', ' ', resultText)
    return resultText
        
        
def CorrectSpelling(inputTokensList):
    correctedTokensList = []
    
    if (inputTokensList is not None) and (len(inputTokensList)!=0):                    
        spell = SpellChecker()
        for word in inputTokensList:
            word = word.lower()
            if word not in spell:
                word = spell.correction(word)
            correctedTokensList.append(word)       
    return correctedTokensList
        
def ReplaceAcronym(inputTokensList, acrDict=None):
    resultTokensList = []
    
    if (inputTokensList is not None) and (len(inputTokensList)!=0): 
        if ((acrDict is not None) and (len(acrDict) != 0)):
            acrDictLowercase = dict((key.lower(), value.lower()) for key, value in acrDict.items())
            resultTokensList = [acrDictLowercase.get(token.lower(), token.lower()) for token in inputTokensList]
        else:
            resultTokensList = inputTokensList
        
    return resultTokensList            
        
def ExpandContractions(inputText):
        resultText = ""
        if inputText != '':
            resultText = contractions.fix(inputText)
        return resultText        

def cleanText( inputText,
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
               fRemovePuncWithinTokens = False
               ):
    if inputText is not None and inputText != "":   
        for function in functionSequence:
            if function == 'RemoveNoise':
                if (fRemoveNoise == True):
                    inputText = RemoveNoise(inputText, 
                                                 removeNoise_fHtmlDecode, 
                                                 removeNoise_fRemoveHyperLinks,
                                                 removeNoise_fRemoveMentions, 
                                                 removeNoise_fRemoveHashtags, 
                                                 removeNoise_RemoveOrReplaceEmoji, 
                                                 removeNoise_fUnicodeToAscii, 
                                                 removeNoise_fRemoveNonAscii)
            if function == 'ExpandContractions':
                if (fExpandContractions == True):
                    inputText = ExpandContractions(inputText)
            if function == 'Normalize':                        
                if (fNormalize == True):
                    inputTokens = WordTokenize(inputText, tokenizationLib)   
                    if (normalizationMethod == 'Stemming'):
                        inputTokens = Stemmize(inputTokens)
                    else:                        
                        inputTokens = Lemmatize(inputTokens, lemmatizationLib)
                    inputText = " ".join(inputTokens)
            if function == 'ReplaceAcronym':   
                if (fReplaceAcronym == True):
                    inputText = ToLowercase(inputText)
                    inputTokens = WordTokenize(inputText, tokenizationLib) 
                    inputTokens= ReplaceAcronym(inputTokens, acronymDict)
                    inputText = " ".join(inputTokens)
            if function == 'CorrectSpelling':
                if (fCorrectSpelling == True):
                    inputTokens = WordTokenize(inputText, tokenizationLib) 
                    inputTokens = CorrectSpelling(inputTokens)
                    inputText = " ".join(inputTokens)
            if function == 'RemoveStopwords':
                if (fRemoveStopwords == True):
                    inputText = ToLowercase(inputText)
                    inputTokens = WordTokenize(inputText, tokenizationLib)
                    inputTokens = RemoveStopwords(inputTokens, stopwordsRemovalLib, stopwordsList, extend_or_replace_stopwordslist)
                    inputText = " ".join(inputTokens)
            if function == 'RemovePunctuation':
                if (fRemovePunctuation == True):
                    inputText = RemovePunctuation(inputText, fRemovePuncWithinTokens)
            if function == 'RemoveNumericTokens':    
                if (fRemoveNumericTokens == True):
                    inputText = RemoveNumericTokens(inputText, removeNumeric_fIncludeSpecialCharacters)
            inputText = ToLowercase(inputText)

    return inputText
        
