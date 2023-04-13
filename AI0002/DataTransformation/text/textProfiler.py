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
import os
import logging
from distutils.util import strtobool
import numpy as np
import pandas as pd
from text import TextProcessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

external_model = None
external_model_type = None

def get_one_true_option(d, default_value):
	if isinstance(d, dict):
		for k,v in d.items():
			if (isinstance(v, str) and v.lower() == 'true') or (isinstance(v, bool) and v == True):
				return k
	return default_value


class textProfiler():
		
	def __init__(self):
		self.log = logging.getLogger('eion')
		self.embedder = None
		self.bert_embedder_size = 0
		
	def textCleaning(self, textCorpus):
		textProcessor = TextProcessing.TextProcessing()
		textCorpus = textProcessor.transform(textCorpus)
		return(textCorpus)
	
	def sentense_encode(self, item):
		return self.model.encode(item,show_progress_bar=False)
		
	def get_embedding_size(self, model, config):
		if model in config.keys():
			config = config[model]
		else:
			config = {}
		model = model.lower()
		if model == 'glove':
			size_map = {'default': 100, '50d': 50, '100d':100, '200d': 200, '300d':300}
			size_enabled = get_one_true_option(config, 'default')
			return size_map[size_enabled]
		elif model == 'fasttext':
			size_map = {'default': 300}
			size_enabled = get_one_true_option(config, 'default')
			return size_map[size_enabled]
		elif model in ['tf_idf', 'countvectors']:
			return int(config.get('dimensions', 200))
		else:   # for word2vec
			return 300
			
		
	def textProfiler(self, conf_json, pipeList, data_path=None):
		cleaning_kwargs = {}
		textCleaning = conf_json.get('textCleaning')
		self.log.info("Text Preprocessing config: ",textCleaning)
		cleaning_kwargs['fRemoveNoise'] = strtobool(textCleaning.get('removeNoise', 'True'))
		cleaning_kwargs['fNormalize'] = strtobool(textCleaning.get('normalize', 'True'))
		cleaning_kwargs['fReplaceAcronym'] = strtobool(textCleaning.get('replaceAcronym', 'False'))
		cleaning_kwargs['fCorrectSpelling'] = strtobool(textCleaning.get('correctSpelling', 'False'))
		cleaning_kwargs['fRemoveStopwords'] = strtobool(textCleaning.get('removeStopwords', 'True'))
		cleaning_kwargs['fRemovePunctuation'] = strtobool(textCleaning.get('removePunctuation', 'True'))
		cleaning_kwargs['fRemoveNumericTokens'] = strtobool(textCleaning.get('removeNumericTokens', 'True'))
		cleaning_kwargs['normalizationMethod'] = get_one_true_option(textCleaning.get('normalizeMethod'),
																	 'lemmatization').capitalize()
		
		removeNoiseConfig = textCleaning.get('removeNoiseConfig')
		if type(removeNoiseConfig) is dict:
			cleaning_kwargs['removeNoise_fHtmlDecode'] = strtobool(removeNoiseConfig.get('decodeHTML', 'True'))
			cleaning_kwargs['removeNoise_fRemoveHyperLinks'] = strtobool(removeNoiseConfig.get('removeHyperLinks', 'True'))
			cleaning_kwargs['removeNoise_fRemoveMentions'] = strtobool(removeNoiseConfig.get('removeMentions', 'True'))
			cleaning_kwargs['removeNoise_fRemoveHashtags'] = strtobool(removeNoiseConfig.get('removeHashtags', 'True'))
			cleaning_kwargs['removeNoise_RemoveOrReplaceEmoji'] = 'remove' if strtobool(removeNoiseConfig.get('removeEmoji', 'True')) else 'replace'
			cleaning_kwargs['removeNoise_fUnicodeToAscii'] = strtobool(removeNoiseConfig.get('unicodeToAscii', 'True'))
			cleaning_kwargs['removeNoise_fRemoveNonAscii'] = strtobool(removeNoiseConfig.get('removeNonAscii', 'True'))

		acronymConfig = textCleaning.get('acronymConfig')
		if type(acronymConfig) is dict:
			cleaning_kwargs['acronymDict'] = acronymConfig.get('acronymDict', None)
		
		stopWordsConfig = textCleaning.get('stopWordsConfig')
		if type(stopWordsConfig) is dict:
			cleaning_kwargs['stopwordsList'] = stopWordsConfig.get('stopwordsList', '[]')
			if isinstance(cleaning_kwargs['stopwordsList'], str):
				if cleaning_kwargs['stopwordsList'] != '[]':
					cleaning_kwargs['stopwordsList'] = cleaning_kwargs['stopwordsList'][1:-1].split(',')
				else:
					cleaning_kwargs['stopwordsList'] = []
			cleaning_kwargs['extend_or_replace_stopwordslist'] = 'replace' if strtobool(stopWordsConfig.get('replace', 'True')) else 'extend'
		removeNumericConfig = textCleaning.get('removeNumericConfig')
		if type(removeNumericConfig) is dict:
			cleaning_kwargs['removeNumeric_fIncludeSpecialCharacters'] = strtobool(removeNumericConfig.get('removeNumeric_IncludeSpecialCharacters', 'True'))

		removePunctuationConfig = textCleaning.get('removePunctuationConfig')
		if type(removePunctuationConfig) is dict:
			cleaning_kwargs['fRemovePuncWithinTokens'] = strtobool(removePunctuationConfig.get('removePuncWithinTokens', 'False'))
		
		cleaning_kwargs['fExpandContractions'] = strtobool(textCleaning.get('expandContractions', 'False'))

		libConfig = textCleaning.get('libConfig')
		if type(libConfig) is dict:
			cleaning_kwargs['tokenizationLib'] = get_one_true_option(libConfig.get('tokenizationLib'), 'nltk')
			cleaning_kwargs['lemmatizationLib'] = get_one_true_option(libConfig.get('lemmatizationLib'), 'nltk')
			cleaning_kwargs['stopwordsRemovalLib'] = get_one_true_option(libConfig.get('stopwordsRemovalLib'), 'nltk')
		if data_path:
			cleaning_kwargs['data_path'] = data_path
		textProcessor = TextProcessing.TextProcessing(**cleaning_kwargs)
		#textCorpus = textProcessor.transform(textCorpus)
		pipeList.append(("TextProcessing",textProcessor))

		textFeatureExtraction = conf_json.get('textFeatureExtraction')
		if strtobool(textFeatureExtraction.get('pos_tags', 'False')):
			pos_tags_lib = get_one_true_option(textFeatureExtraction.get('pos_tags_lib'), 'nltk')
			posTagger = TextProcessing.PosTagging( pos_tags_lib, data_path)
			#textCorpus = posTagger.transform(textCorpus)
			pipeList.append(("posTagger",posTagger))
		ngram_min = 1
		ngram_max = 1
		if strtobool(textFeatureExtraction.get('n_grams', 'False')):
			n_grams_config = textFeatureExtraction.get("n_grams_config")
			ngram_min = int(n_grams_config.get('min_n', 1))
			ngram_max = int(n_grams_config.get('max_n', 1))
			if (ngram_min < 1) or ngram_min > ngram_max:
				ngram_min = 1
				ngram_max = 1
				invalidNgramWarning = 'WARNING : invalid ngram config.\nUsing the default values min_n={}, max_n={}'.format(ngram_min, ngram_max)
				self.log.info(invalidNgramWarning)
		ngram_range_tuple = (ngram_min, ngram_max)
		textConversionMethod = conf_json.get('textConversionMethod')
		conversion_method = get_one_true_option(textConversionMethod, None)
		embedding_size_config = conf_json.get('embeddingSize', {})
		embedding_size = self.get_embedding_size(conversion_method, embedding_size_config)
		if conversion_method.lower() == "countvectors":
			vectorizer = TextProcessing.ExtractFeatureCountVectors( ngram_range=ngram_range_tuple, max_features=embedding_size)
			pipeList.append(("vectorizer",vectorizer))
			self.log.info('----------> Conversion Method: CountVectors')
		elif conversion_method.lower() in ["fasttext","glove"]:
			embedding_method = conversion_method
			wordEmbeddingVecotrizer = TextProcessing.wordEmbedding(embedding_method, embedding_size)
			#wordEmbeddingVecotrizer.checkAndDownloadPretrainedModel()
			pipeList.append(("vectorizer",wordEmbeddingVecotrizer))
			self.log.info('----------> Conversion Method: '+str(conversion_method))
		elif conversion_method.lower() == "sentencetransformer":
			from sentence_transformers import SentenceTransformer
			embedding_pretrained = {'model':'sentence-transformers/msmarco-distilroberta-base-v2','size': 768}
			self.bert_embedder_size = embedding_pretrained['size']
			self.model = SentenceTransformer(embedding_pretrained['model'])
			self.embedder = FunctionTransformer(self.sentense_encode, feature_names_out = self.sentence_transformer_output)
			pipeList.append(("vectorizer",self.embedder))
			self.log.info('----------> Conversion Method: SentenceTransformer')
		elif conversion_method.lower() == 'tf_idf':
			vectorizer = TextProcessing.ExtractFeatureTfIdfVectors(ngram_range=ngram_range_tuple)
			pipeList.append(("vectorizer",vectorizer))
			self.log.info('----------> Conversion Method: TF_IDF')
		else:
			df1 = pd.DataFrame()
			#df1['tokenize'] = textCorpus
			self.log.info('----------> Conversion Method: NA')            
		return pipeList

	def sentence_transformer_output(self, transformer, names=None):
		return [str(x) for x in range(self.bert_embedder_size)]
		

class textCombine(TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		if X.shape[1] > 1:
			return np.array([" ".join(i) for i in X])
		else:
			if isinstance(X, np.ndarray):
				return np.ndarray.flatten(X)
			else:    
				return X

def get_pretrained_model_path():
    try:
        from appbe.dataPath import DATA_DIR
        modelsPath = Path(DATA_DIR)/'PreTrainedModels'/'TextProcessing'
    except:
        modelsPath = Path('aion')/'PreTrainedModels'/'TextProcessing'
    
    if not modelsPath.exists():
        modelsPath.mkdir(parents=True, exist_ok=True)
    return modelsPath
    
def set_pretrained_model(pipe):
    from text.Embedding import load_pretrained
    import importlib.util
    global external_model
    global external_model_type
    params = pipe.get_params()
    model_name = params.get('text_process__vectorizer__preTrainedModel', None)
    if model_name and model_name.lower() in ['fasttext','glove'] and not external_model:
        if model_name == 'fasttext' and importlib.util.find_spec('fasttext'):
            import fasttext
            import fasttext.util
            cwd = os.getcwd()
            os.chdir(get_pretrained_model_path())
            fasttext.util.download_model('en', if_exists='ignore')
            external_model = fasttext.load_model('cc.en.300.bin')
            os.chdir(cwd)
            external_model_type = 'binary'
            print('loaded fasttext binary')
        else:
            model_path = TextProcessing.checkAndDownloadPretrainedModel(model_name)
            embed_size, external_model = load_pretrained(model_path)
            external_model_type = 'vector'
            print(f'loaded {model_name} vector')
        pipe.set_params(text_process__vectorizer__external_model = external_model)
        pipe.set_params(text_process__vectorizer__external_model_type = external_model_type)

def reset_pretrained_model(pipe, clear_mem=True):
    global external_model
    global external_model_type
    params = pipe.get_params()
    is_external_model = params.get('text_process__vectorizer__external_model', None)
    if (isinstance(is_external_model, pd.DataFrame) and not is_external_model.empty) or is_external_model:
        pipe.set_params(text_process__vectorizer__external_model = None)
        pipe.set_params(text_process__vectorizer__external_model_type = None)
        if clear_mem:
            external_model = None

def release_pretrained_model():
    global external_model
    global external_model_type
    external_model = None
    external_model_type = None
    