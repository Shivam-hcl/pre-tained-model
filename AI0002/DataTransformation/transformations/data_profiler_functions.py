import numpy as np
default_config = {
    'misValueRatio': '1.0',
    'numericFeatureRatio': '1.0',
    'categoryMaxLabel': '20',
    'str_to_cat_len_max': 10
}

target_encoding_method_change = {'targetencoding': 'labelencoding'}

supported_method = {
    'fillNa':
        {
            'categorical' : ['mode','zero','na'],
            'numeric' : ['median','mean','knnimputer','zero','drop','na'],
        },
    'categoryEncoding': ['labelencoding','targetencoding','onehotencoding','na','none'],
    'normalization': ['standardscaler','minmax','lognormal', 'na','none'],
    'outlier_column_wise': ['iqr','zscore', 'disable'],
    'outlierOperation': ['dropdata', 'average', 'nochange']
    }
    
def findiqrOutlier(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    index = ~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR)))
    return index
    
def findzscoreOutlier(df):
    z = np.abs(scipy.stats.zscore(df))
    index = (z < 3)
    return index
    
def findiforestOutlier(df):
    from sklearn.ensemble import IsolationForest
    isolation_forest = IsolationForest(n_estimators=100)
    isolation_forest.fit(df)
    y_pred_train = isolation_forest.predict(df)
    return y_pred_train == 1

def get_one_true_option(d, default_value=None):
	if isinstance(d, dict):
		for k,v in d.items():
			if (isinstance(v, str) and v.lower() == 'true') or (isinstance(v, bool) and v == True):
				return k
	return default_value

def get_boolean(value):
    if (isinstance(value, str) and value.lower() == 'true') or (isinstance(value, bool) and value == True):
        return True
    else:
        return False
        
def recommenderStartProfiler(self,modelFeatures):
    try:
        self.log.info('----------> FillNA:0')
        self.data = self.data.fillna(value=0)
        self.log.info('Status:-        !... Missing value treatment done')
        self.log.info('----------> Remove Empty Row')
        self.data = self.data.dropna(axis=0,how='all')
        self.log.info('Status:-        !... Empty feature treatment done')                      
        userId,itemId,rating = modelFeatures.split(',')
        self.data[itemId] = self.data[itemId].astype(np.int32)
        self.data[userId] = self.data[userId].astype(np.int32)
        self.data[rating] = self.data[rating].astype(np.float32)
        return self.data
    except Exception as inst:
        self.log.info("Error: dataProfiler failed "+str(inst))
        return(self.data)

def folderPreprocessing(self,folderlocation,folderdetails,deployLocation):
    try:
        dataset_directory = Path(folderlocation)
        dataset_csv_file = dataset_directory/folderdetails['label_csv_file_name']
        tfrecord_directory = Path(deployLocation)/'Video_TFRecord'
        from savp import PreprocessSAVP
        import csv
        csvfile = open(dataset_csv_file, newline='')
        csv_reader = csv.DictReader(csvfile)
        PreprocessSAVP(dataset_directory,csv_reader,tfrecord_directory)
        dataColumns = list(self.data.columns)
        VideoProcessing = True
        return dataColumns,VideoProcessing,tfrecord_directory
    except Exception as inst:
        self.log.info("Error: dataProfiler failed "+str(inst))

def textSimilarityStartProfiler(self, doc_col_1, doc_col_2):
    import os
    try:
        features = [doc_col_1, doc_col_2]
        pipe = None
        dataColumns = list(self.data.columns)
        self.numofCols = self.data.shape[1]
        self.numOfRows = self.data.shape[0]
        from transformations.textProfiler import textProfiler            
        
        self.log.info('-------> Execute Fill NA With Empty String')
        self.data = self.data.fillna(value=" ")
        self.log.info('Status:- |... Missing value treatment done')
        self.data[doc_col_1] = textProfiler().textCleaning(self.data[doc_col_1])
        self.data[doc_col_2] = textProfiler().textCleaning(self.data[doc_col_2])
        self.log.info('-------> Concatenate: ' + doc_col_1 + ' ' + doc_col_2)
        self.data['text'] = self.data[[doc_col_1, doc_col_2]].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        from tensorflow.keras.preprocessing.text import Tokenizer
        pipe = Tokenizer()
        pipe.fit_on_texts(self.data['text'].values)
        self.log.info('-------> Tokenizer: Fit on  Concatenate Field')
        self.log.info('Status:- |... Tokenizer the text')
        self.data[doc_col_1] = self.data[doc_col_1].astype(str)
        self.data[doc_col_1] = self.data[doc_col_1].astype(str)
        return (self.data, pipe, self.target_name, features)
    except Exception as inst:
        self.log.info("StartProfiler failed " + str(inst))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        self.log.info(str(exc_type) + ' ' + str(fname) + ' ' + str(exc_tb.tb_lineno))

def set_features(features,profiler=None):
    if profiler:
        features = [x for x in features if x not in profiler.added_features]
        return features + profiler.text_feature
    return features