'''
*
* =============================================================================
* COPYRIGHT NOTICE
* =============================================================================
*  @ Copyright HCL Technologies Ltd. 2021, 2022,2023
* Proprietary and confidential. All information contained herein is, and
* remains the property of HCL Technologies Limited. Copying or reproducing the
* contents of this file, via any medium is strictly prohibited unless prior
* written permission is obtained from HCL Technologies Limited.
*
'''
import io
import json
import logging
import pandas as pd
import sys
import numpy as np
from pathlib import Path
from word2number import w2n
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.ensemble import IsolationForest
from category_encoders import TargetEncoder
import scipy
import transformations.data_profiler_functions as cs
from text.textProfiler import set_pretrained_model

if 'AION' in sys.modules:
    try:
        from appbe.app_config import DEBUG_ENABLED
    except:
        DEBUG_ENABLED = False
else:
    DEBUG_ENABLED = False
print(f'Debug Enabled: {DEBUG_ENABLED}')

log_suffix = f'[{Path(__file__).stem}] '

class profiler():

    def __init__(self, xtrain, ytrain=None, target=None, encode_target = True, config={}, keep_unprocessed=[],data_path=None,log=None):
        if not isinstance(xtrain, pd.DataFrame):
            raise ValueError(f'{log_suffix}supported data type is pandas.DataFrame but provide data is of {type(xtrain)} type')
        if xtrain.empty:
            raise ValueError(f'{log_suffix}Data frame is empty')
        if target and target in xtrain.columns:
            self.target = xtrain[target]
            xtrain.drop(target, axis=1, inplace=True)
            self.target_name = target
        elif ytrain:
            self.target = ytrain
            self.target_name = 'target'
        else:
            self.target = pd.Series()
            self.target_name = None
        self.data_path = data_path
        self.encode_target = encode_target
        self.label_encoder = None
        keep_unprocessed = [x for x in keep_unprocessed if x in xtrain.columns]
        if keep_unprocessed:
            self.unprocessed = xtrain[keep_unprocessed]
            self.data = xtrain.drop(keep_unprocessed, axis=1)
        else:
            self.data = xtrain        
            self.unprocessed = pd.DataFrame()
        self.colm_type = {}
        for colm, infer_type in zip(self.data.columns, self.data.dtypes):
            self.colm_type[colm] = infer_type
        self.numeric_feature = []
        self.cat_feature = []
        self.text_feature = []
        self.wordToNumericFeatures = []
        self.added_features = []
        self.pipeline = []
        self.dropped_features = {}
        self.train_features_type={}
        self.__update_type()
        self.config = config
        self.featureDict = config.get('featureDict', [])
        self.output_columns = []
        self.feature_expender = []
        self.text_to_num = {}
        if log:
            self.log = log
        else:
            self.log = logging.getLogger('eion')
        self.type_conversion = {}

    def log_dataframe(self, msg=None):                    
        buffer = io.StringIO()                    
        self.data.info(buf=buffer)                    
        if msg:                    
            log_text = f'Data frame after {msg}:'                    
        else:                    
            log_text = 'Data frame:'                    
        log_text += '\n\t'+str(self.data.head(2)).replace('\n','\n\t')                    
        log_text += ('\n\t' + buffer.getvalue().replace('\n','\n\t'))                    
        self.log.info(log_text)

    def transform(self):
        if self.is_target_available():
            if self.target_name:
                self.log.info(f"Target feature name: '{self.target_name}'")
            self.log.info(f"Target feature size: {len(self.target)}")
        else:
            self.log.info(f"Target feature not present")
        self.log_dataframe()
        try:
            self.process()
        except Exception as e:
            
            self.log.error(e, exc_info=True)
            raise
        pipe = FeatureUnion(self.pipeline)
        try:
            set_pretrained_model(pipe)
            conversion_method = self.get_conversion_method()
            process_data = pipe.fit_transform(self.data, y=self.target)
            # save for testing
            if DEBUG_ENABLED:
                if isinstance(process_data, scipy.sparse.spmatrix):
                    process_data = process_data.toarray()        
                df = pd.DataFrame(process_data)
                df.to_csv('debug_preprocessed.csv', index=False)
            if self.text_feature and conversion_method == 'tf_idf':
                n_size = self.get_tf_idf_output_size( pipe)
                dimensions = self.get_tf_idf_dimensions()
                if n_size < dimensions:
                    dimensions = n_size
                from sklearn.decomposition import TruncatedSVD
                reducer = TruncatedSVD( n_components = dimensions)
                reduced_data = reducer.fit_transform( process_data[:,-n_size:])
                text_process_idx = [t[0] for t in pipe.transformer_list].index('text_process')
                pipe.transformer_list[text_process_idx][1].steps.append(('feature_reducer',reducer))
                if isinstance(process_data, scipy.sparse.spmatrix):
                    process_data = process_data.toarray()
                process_data = np.concatenate((process_data[:,:-n_size], reduced_data), axis=1)
                last_step = self.feature_expender.pop()
                self.feature_expender.append({'feature_reducer':list(last_step.values())[0]})
                    
        except EOFError as e:
            if "Compressed file ended before the end-of-stream marker was reached" in str(e):
                raise EOFError('Pretrained model is not downloaded properly')

        self.update_output_features_names(pipe)
        if isinstance(process_data, scipy.sparse.spmatrix):
            process_data = process_data.toarray()        
        df = pd.DataFrame(process_data, columns=self.output_columns)
        
        if self.is_target_available() and self.target_name:
            df[self.target_name] = self.target
        if not self.unprocessed.empty:
            df[self.unprocessed.columns] = self.unprocessed
        self.log_numerical_fill()
        self.log_categorical_fill()
        self.log_normalization()
        return df, pipe, self.label_encoder
    
    def log_type_conversion(self):
        if self.log:
            self.log.info('-----------  Inspecting Features -----------')
            self.log.info('-----------  Type Conversion -----------')
            count = 0
            for k, v in self.type_conversion.items():
                if v[0] != v[1]:
                    self.log.info(f'{k} -> from {v[0]} to {v[1]} : {v[2]}')
            self.log.info('Status:- |... Feature inspection done')
    
    def check_config(self):
        removeDuplicate = self.config.get('removeDuplicate', False)
        self.config['removeDuplicate'] = cs.get_boolean(removeDuplicate)
        self.config['misValueRatio'] = float(self.config.get('misValueRatio', cs.default_config['misValueRatio']))
        self.config['numericFeatureRatio'] = float(self.config.get('numericFeatureRatio', cs.default_config['numericFeatureRatio']))
        self.config['categoryMaxLabel'] = int(self.config.get('categoryMaxLabel', cs.default_config['categoryMaxLabel']))
        featureDict = self.config.get('featureDict', [])
        if isinstance(featureDict, dict):
            self.config['featureDict'] = []
        if isinstance(featureDict, str):
            self.config['featureDict'] = []
                
    def process(self):
        #remove duplicate not required at the time of prediction
        self.check_config()
        self.remove_constant_feature()
        self.remove_empty_feature(self.config['misValueRatio'])
        self.remove_index_features()
        if self.config['removeDuplicate']:
            self.drop_duplicate()
        self.check_categorical_features()
        self.string_to_numeric()
        self.process_target()
        self.train_features_type = {k:v for k,v in zip(self.data.columns, self.data.dtypes)}
        self.parse_process_step_config()
        self.process_drop_fillna()
        #self.log_type_conversion()
        self.update_num_fill_dict()
        #print(self.num_fill_method_dict)
        self.update_cat_fill_dict()
        self.create_pipeline()
        self.text_pipeline(self.config)
        self.apply_outlier()
        self.log.info(self.process_method)
        #self.log.info(self.pipeline)
    
    def is_target_available(self):
        return (isinstance(self.target, pd.Series) and not self.target.empty) or len(self.target)
    
    def process_target(self, operation='encode', arg=None):
        if self.is_target_available():
            # drop null values
            self.__update_index( self.target.notna(), 'target')
            if self.encode_target:
                self.label_encoder = LabelEncoder()
                self.target = self.label_encoder.fit_transform(self.target)
                return self.label_encoder
        return None
        
    def is_target_column(self, column):
        return column == self.target_name
    
    def fill_default_steps(self):
        
        num_fill_method = cs.get_one_true_option(self.config.get('numericalFillMethod',None))
        normalization_method = cs.get_one_true_option(self.config.get('normalization',None))
        for colm in self.numeric_feature:
            if num_fill_method:
                self.fill_missing_value_method(colm, num_fill_method.lower())
            if normalization_method:
                self.fill_normalizer_method(colm, normalization_method.lower())
        
        cat_fill_method = cs.get_one_true_option(self.config.get('categoricalFillMethod',None))
        cat_encode_method = cs.get_one_true_option(self.config.get('categoryEncoding',None))
        for colm in self.cat_feature:
            if cat_fill_method:
                self.fill_missing_value_method(colm, cat_fill_method.lower())
            if cat_encode_method:
                self.fill_encoder_value_method(colm, cat_encode_method.lower(), default=True)
    
    def parse_process_step_config(self):
        self.process_method = {}
        user_provided_data_type = {}
        for feat_conf in self.featureDict:
            colm = feat_conf.get('feature', '')
            if not self.is_target_column(colm):
                if colm in self.data.columns:
                    user_provided_data_type[colm] = feat_conf['type']
        if user_provided_data_type:
            self.update_user_provided_type(user_provided_data_type)
                    
        self.fill_default_steps()
        for feat_conf in self.featureDict:
            colm = feat_conf.get('feature', '')
            if not self.is_target_column(colm):
                if colm in self.data.columns:
                    if feat_conf.get('fillMethod', None):
                        self.fill_missing_value_method(colm, feat_conf['fillMethod'].lower())
                    if feat_conf.get('categoryEncoding', None):
                        self.fill_encoder_value_method(colm, feat_conf['categoryEncoding'].lower())
                    if feat_conf.get('normalization', None):
                        self.fill_normalizer_method(colm, feat_conf['normalization'].lower())
                    if feat_conf.get('outlier', None):
                        self.fill_outlier_method(colm, feat_conf['outlier'].lower())
                    if feat_conf.get('outlierOperation', None):
                        self.fill_outlier_process(colm, feat_conf['outlierOperation'].lower())
        
    
    def get_tf_idf_dimensions(self):
        dim = cs.get_one_true_option(self.config.get('embeddingSize',{}).get('TF_IDF',{}), 'default')
        return {'default': 300, '50d':50, '100d':100, '200d':200, '300d':300}[dim]

    def get_tf_idf_output_size(self, pipe):
        start_index = {}
        for feat_expender in self.feature_expender:
            if feat_expender:
                step_name = list(feat_expender.keys())[0]
                index = list(feat_expender.values())[0]
                for transformer_step in pipe.transformer_list:
                    if transformer_step[1].steps[-1][0] in step_name:
                        start_index[index] = {transformer_step[1].steps[-1][0]: transformer_step[1].steps[-1][1].get_feature_names_out()}    
        if start_index:
            for key,value in start_index.items():
                for k,v in value.items():
                    if k == 'vectorizer':
                        return len(v)
        return 0

    def update_output_features_names(self, pipe):
        columns = self.output_columns
        start_index = {}
        index_shifter = 0
        for feat_expender in self.feature_expender:
            if feat_expender:
                step_name = list(feat_expender.keys())[0]
                for key,value in start_index.items():
                    for k,v in value.items():
                        index_shifter += len(v)
                index = list(feat_expender.values())[0]
                for transformer_step in pipe.transformer_list:
                    if transformer_step[1].steps[-1][0] in step_name:
                        start_index[index + index_shifter] = {transformer_step[1].steps[-1][0]: transformer_step[1].steps[-1][1].get_feature_names_out()}
        #print(start_index)
        if start_index:
            for key,value in start_index.items():
                for k,v in value.items():
                    if k == 'vectorizer':
                        v = [f'{x}_vect' for x in v]    
                    self.output_columns[key:key] = v
                    self.added_features = [*self.added_features, *v]
                    
                    
    def text_pipeline(self, conf_json):

        if self.text_feature:
            from text.textProfiler import textProfiler
            from text.textProfiler import textCombine
            pipeList = []
            text_pipe = Pipeline([ 
                ('selector', ColumnTransformer([
                        ("selector", "passthrough", self.text_feature)
                        ], remainder="drop")),
                ("text_fillNa",SimpleImputer(strategy='constant', fill_value='')),
                ("merge_text_feature", textCombine())])
            obj = textProfiler()
            pipeList = obj.textProfiler(conf_json, pipeList, self.data_path)
            last_step = "merge_text_feature"
            for pipe_elem in pipeList:
                text_pipe.steps.append((pipe_elem[0], pipe_elem[1]))
                last_step = pipe_elem[0]
            text_transformer = ('text_process', text_pipe)
            self.pipeline.append(text_transformer)
            self.feature_expender.append({last_step:len(self.output_columns)})
            
    def create_pipeline(self):
        num_pipe = {}
        for k,v in self.num_fill_method_dict.items():
            for k1,v1 in v.items():
                if k1 and k1 != 'none':
                    num_pipe[f'{k}_{k1}'] = Pipeline([
                        ('selector', ColumnTransformer([
                                ("selector", "passthrough", v1)
                                ], remainder="drop")),
                        (k, self.get_num_imputer(k)),
                        (k1, self.get_num_scaler(k1))
                    ])
                else:
                    num_pipe[f'{k}_{k1}'] = Pipeline([
                        ('selector', ColumnTransformer([
                                ("selector", "passthrough", v1)
                                ], remainder="drop")),
                        (k, self.get_num_imputer(k))
                    ])
                self.output_columns.extend(v1)
        cat_pipe = {}
        for k,v in self.cat_fill_method_dict.items():
            for k1,v1 in v.items():
                cat_pipe[f'{k}_{k1}'] = Pipeline([
                    ('selector', ColumnTransformer([
                            ("selector", "passthrough", v1)
                            ], remainder="drop")),
                    (k, self.get_cat_imputer(k)),
                    (k1, self.get_cat_encoder(k1))
                ])                   
                if k1 not in ['onehotencoding']:
                    self.output_columns.extend(v1)
                else:
                    self.feature_expender.append({k1:len(self.output_columns)})
        for key, pipe in num_pipe.items():            
            self.pipeline.append((key, pipe))
        for key, pipe in cat_pipe.items():            
            self.pipeline.append((key, pipe))
    
    "Drop: feature during training but replace with zero during prediction "
    def process_drop_fillna(self):
        drop_column = []
        if 'numFill' in self.process_method.keys():
            for col, method in self.process_method['numFill'].items():
                if method == 'drop':
                    self.process_method['numFill'][col] = 'zero'
                    drop_column.append(col)
        if 'catFill' in self.process_method.keys():
            for col, method in self.process_method['catFill'].items():
                if method == 'drop':
                    self.process_method['catFill'][col] = 'zero'
                    drop_column.append(col)
        if drop_column:
            self.data.dropna(subset=drop_column, inplace=True)

    def update_num_fill_dict(self):
        self.num_fill_method_dict = {}
        if 'numFill' in self.process_method.keys():
            for f in cs.supported_method['fillNa']['numeric']:
                self.num_fill_method_dict[f] = {}
                for en in cs.supported_method['normalization']:
                    self.num_fill_method_dict[f][en] = []
                    for col in self.numeric_feature:
                        numFillDict = self.process_method.get('numFill',{})
                        normalizationDict = self.process_method.get('normalization',{})
                        if f == numFillDict.get(col, '') and en == normalizationDict.get(col,''):
                            self.num_fill_method_dict[f][en].append(col)
                    if not self.num_fill_method_dict[f][en] :
                        del self.num_fill_method_dict[f][en]
                if not self.num_fill_method_dict[f]:
                    del self.num_fill_method_dict[f]

    def update_cat_fill_dict(self):
        self.cat_fill_method_dict = {}
        if 'catFill' in self.process_method.keys():
            for f in cs.supported_method['fillNa']['categorical']:
                self.cat_fill_method_dict[f] = {}
                for en in cs.supported_method['categoryEncoding']:
                    self.cat_fill_method_dict[f][en] = []
                    for col in self.cat_feature:
                        catFillDict = self.process_method.get('catFill',{})
                        catEncoderDict = self.process_method.get('catEncoder',{})
                        if f == catFillDict.get(col, '') and en == catEncoderDict.get(col,''):
                            self.cat_fill_method_dict[f][en].append(col)
                    if not self.cat_fill_method_dict[f][en] :
                        del self.cat_fill_method_dict[f][en]
                if not self.cat_fill_method_dict[f]:
                    del self.cat_fill_method_dict[f]


    def __update_type(self):
        self.numeric_feature = self.data.select_dtypes(include='number').columns.tolist()
        self.cat_feature = self.data.select_dtypes(include='category').columns.tolist()
        self.date_time = self.data.select_dtypes(include='datetime').columns.tolist()
        self.text_feature = self.data.select_dtypes(include='object').columns.tolist()
        
    def update_user_provided_type(self, data_types):
        allowed_types = ['numerical','categorical', 'text','date','index']
        type_mapping = {'numerical': np.dtype('float'), 'float': np.dtype('float'),'categorical': 'category', 'text':np.dtype('object'),'date':'datetime64[ns]','index': np.dtype('int64'),}
        mapped_type = {k:type_mapping[v] for k,v in data_types.items()}
        #self.log.info(mapped_type)
        self.update_type(mapped_type, 'user provided data type')
        
    def get_type(self, as_list=False):
        if as_list:
            return [self.colm_type.values()]
        else:
            return self.colm_type
    
    def update_type(self, data_types={}, reason=''):
        invalid_features = [x for x in data_types.keys() if x not in self.data.columns]
        if invalid_features:
            valid_feat = list(set(data_types.keys()) - set(invalid_features))
            valid_feat_type = {k:v for k,v in data_types if k in valid_feat}
        else:
            valid_feat_type = data_types
        for k,v in valid_feat_type.items():
            if v != self.colm_type[k].name:
                try:
                    self.data.astype({k:v})
                    self.colm_type.update({k:self.data[k].dtype})
                    self.type_conversion[k] = (self.colm_type[k] , v, 'Done', reason)
                except:
                    self.type_conversion[k] = (self.colm_type[k] , v, 'Fail', reason)
        self.data = self.data.astype(valid_feat_type)
        self.__update_type()
        
    def string_to_numeric(self):
        def to_number(x):
            try:
                return w2n.word_to_num(x)
            except:
                return np.nan
        for col in self.text_feature:
            col_values = self.data[col].copy()
            col_values = pd.to_numeric(col_values, errors='coerce')
            if col_values.count() >= (self.config['numericFeatureRatio'] * len(col_values)):
                self.text_to_num[col] = 'float64'
                self.wordToNumericFeatures.append(col)
        if self.text_to_num:
            columns = list(self.text_to_num.keys())
            self.data[columns] = self.data[columns].apply(lambda x: to_number(x))
            self.update_type(self.text_to_num)
        self.log.info('-----------  Inspecting Features -----------')
        for col in self.text_feature:
            self.log.info(f'-------> Feature : {col}')
            if col in self.text_to_num:
                self.log.info('----------> Numeric Status :Yes')
                self.log.info('----------> Data Type Converting to numeric :Yes')
            else:
                self.log.info('----------> Numeric Status :No')
        self.log.info(f'\nStatus:- |... Feature inspection done for numeric data: {len(self.text_to_num)} feature(s) converted to numeric')
        self.log.info(f'\nStatus:- |... Feature word to numeric treatment done: {self.text_to_num}')
        self.log.info('-----------   Inspecting Features End -----------')
            
    def check_categorical_features(self):
        num_data = self.data.select_dtypes(include='number')
        num_data_unique = num_data.nunique()
        num_to_cat_col = {}
        for i, value in enumerate(num_data_unique):
            if value < self.config['categoryMaxLabel']:
                num_to_cat_col[num_data_unique.index[i]] = 'category'
        if num_to_cat_col:
            self.update_type(num_to_cat_col, 'numerical to categorical')
        str_to_cat_col = {}
        str_data = self.data.select_dtypes(include='object')
        str_data_unique = str_data.nunique()
        for i, value in enumerate(str_data_unique):
            if value < self.config['categoryMaxLabel']:
                str_to_cat_col[str_data_unique.index[i]] = 'category'
        for colm in str_data.columns:
            if self.data[colm].str.len().max() < cs.default_config['str_to_cat_len_max']:
                str_to_cat_col[colm] = 'category'
        if str_to_cat_col:
            self.update_type(str_to_cat_col, 'text to categorical')        
        
    def drop_features(self, features=[], reason='unspecified'):
        if isinstance(features, str):
            features = [features]
        feat_to_remove = [x for x in features if x in self.data.columns]
        if feat_to_remove:
            self.data.drop(feat_to_remove, axis=1, inplace=True)
            for feat in feat_to_remove:
                self.dropped_features[feat] = reason        
            self.log_drop_feature(feat_to_remove, reason)
            self.__update_type()
    
    def __update_index(self, indices, reason=''):
        if isinstance(indices, (bool, pd.core.series.Series)) and len(indices) == len(self.data):
            self.data = self.data[indices]
            self.data.reset_index(drop=True, inplace=True)
            if self.is_target_available():
                self.target = self.target[indices]
                if isinstance(self.target, pd.Series):
                    self.target.reset_index(drop=True, inplace=True)
            if not self.unprocessed.empty:
                self.unprocessed   = self.unprocessed[indices]
                self.unprocessed.reset_index(drop=True, inplace=True)
            self.log_update_index((indices == False).sum(), reason)
    
    def drop_duplicate(self):
        index = self.data.duplicated(keep='first')
        self.__update_index( ~index, reason='duplicate')

    def log_drop_feature(self, columns, reason):
        self.log.info(f'---------- Dropping {reason} features ----------')
        self.log.info(f'\nStatus:- |... {reason} feature treatment done: {len(columns)} {reason} feature(s) found')
        self.log.info(f'-------> Drop Features: {columns}')
        self.log.info(f'Data Frame Shape After Dropping (Rows,Columns): {self.data.shape}')
    
    def log_update_index(self,count, reason):
        if count:
            if reason == 'target':
                self.log.info('-------> Null Target Rows Drop:')
                self.log.info(f'------->     Dropped rows count: {count}')
            elif reason == 'duplicate':
                self.log.info('-------> Duplicate Rows Drop:')
                self.log.info(f'------->     Dropped rows count: {count}')
            elif reason == 'outlier':
                self.log.info(f'------->     Dropped rows count: {count}')
                self.log.info('Status:- |... Outlier treatment done')
            self.log.info(f'------->     Data Frame Shape After Dropping samples(Rows,Columns): {self.data.shape}')
    
    def log_normalization(self):
        if self.process_method.get('normalization', None):
            self.log.info(f'\nStatus:- !... Normalization treatment done')
            for method in cs.supported_method['normalization']:
                cols = []
                for col, m in self.process_method['normalization'].items():
                    if m == method:
                        cols.append(col)
                if cols and method != 'none':
                    self.log.info(f'Running {method} on features: {cols}')

    def log_numerical_fill(self):
        if self.process_method.get('numFill', None):
            self.log.info(f'\nStatus:- !... Fillna for numeric feature done')
            for method in cs.supported_method['fillNa']['numeric']:
                cols = []
                for col, m in self.process_method['numFill'].items():
                    if m == method:
                        cols.append(col)
                if cols:
                    self.log.info(f'-------> Running {method} on features: {cols}')

    def log_categorical_fill(self):
        if self.process_method.get('catFill', None):
            self.log.info(f'\nStatus:- !... FillNa for categorical feature done')
            for method in cs.supported_method['fillNa']['categorical']:
                cols = []
                for col, m in self.process_method['catFill'].items():
                    if m == method:
                        cols.append(col)
                if cols:
                    self.log.info(f'-------> Running {method} on features: {cols}')
    
    
    def remove_constant_feature(self):
        unique_values = self.data.nunique()
        constant_features = []
        for i, value in enumerate(unique_values):
            if value == 1:
                constant_features.append(unique_values.index[i])
        if constant_features:
            self.drop_features(constant_features, "constant")
        
    def remove_empty_feature(self, misval_ratio=1.0):
        missing_ratio = self.data.isnull().sum() / len(self.data)
        missing_ratio = {k:v for k,v in zip(self.data.columns, missing_ratio)}
        empty_features = [k for k,v in missing_ratio.items() if v > misval_ratio]
        if empty_features:
            self.drop_features(empty_features, "empty")

    def remove_index_features(self):
        index_feature = []
        
        for feat in self.numeric_feature:
            if self.data[feat].nunique() == len(self.data):
                if (self.data[feat].sum()- sum(self.data.index) == (self.data.iloc[0][feat]-self.data.index[0])*len(self.data)):
                    index_feature.append(feat)
        self.drop_features(index_feature, "index")

    def fill_missing_value_method(self, colm, method):
        if colm in self.numeric_feature:
            if method in cs.supported_method['fillNa']['numeric']:
                if 'numFill' not in self.process_method.keys():
                    self.process_method['numFill'] = {}
                if method == 'na' and self.process_method['numFill'].get(colm, None):
                    pass # don't overwrite
                else:
                    self.process_method['numFill'][colm] = method
        if colm in self.cat_feature:
            if method in cs.supported_method['fillNa']['categorical']:
                if 'catFill' not in self.process_method.keys():
                    self.process_method['catFill'] = {}
                if method == 'na' and self.process_method['catFill'].get(colm, None):
                    pass
                else:
                    self.process_method['catFill'][colm] = method
    
    def check_encoding_method(self, method, colm,default=False):
        if not self.is_target_available() and (method.lower() == list(cs.target_encoding_method_change.keys())[0]):
            method = cs.target_encoding_method_change[method.lower()]
            if default:
                self.log.info(f"Applying Label encoding instead of Target encoding on feature '{colm}' as target feature is not present")
        return method
    
    def fill_encoder_value_method(self,colm, method, default=False):
        if colm in self.cat_feature:
            if method.lower() in cs.supported_method['categoryEncoding']:
                if 'catEncoder' not in self.process_method.keys():
                    self.process_method['catEncoder'] = {}
                if method == 'na' and self.process_method['catEncoder'].get(colm, None):
                    pass
                else:
                    self.process_method['catEncoder'][colm] = self.check_encoding_method(method, colm,default)
            else:
                self.log.info(f"-------> categorical encoding method '{method}' is not supported. supported methods are {cs.supported_method['categoryEncoding']}")

    def fill_normalizer_method(self,colm, method):
        if colm in self.numeric_feature:
            if method in cs.supported_method['normalization']:
                if 'normalization' not in self.process_method.keys():
                    self.process_method['normalization'] = {}
                if (method == 'na' or method == 'none')  and self.process_method['normalization'].get(colm, None):
                    pass
                else:
                    self.process_method['normalization'][colm] = method
            else:
                self.log.info(f"-------> Normalization method '{method}' is not supported. supported methods are {cs.supported_method['normalization']}")

    def apply_outlier(self):
        inlier_indice = np.array([True] * len(self.data))
        if self.process_method.get('outlier', None):
            self.log.info('-------> Feature wise outlier detection:')
            for k,v in self.process_method['outlier'].items():
                if k in self.numeric_feature:
                    if v == 'iqr':
                        index = cs.findiqrOutlier(self.data[k])
                    elif v == 'zscore':
                        index = cs.findzscoreOutlier(self.data[k])
                    elif v == 'disable':
                        index = None
                    if k in self.process_method['outlierOperation'].keys():
                        if self.process_method['outlierOperation'][k] == 'dropdata':
                            inlier_indice = np.logical_and(inlier_indice, index)
                        elif self.process_method['outlierOperation'][k] == 'average':
                            mean = self.data[k].mean()
                            index = ~index
                            self.data.loc[index,[k]] = mean
                            self.log.info(f'------->     {k}: Replaced by Mean {mean}: total replacement {index.sum()}')
                        elif self.process_method['outlierOperation'][k] == 'nochange' and v != 'disable':
                            self.log.info(f'------->     Total outliers in "{k}": {(~index).sum()}')
        if self.config.get('outlierDetection',None):
            if self.config['outlierDetection'].get('IsolationForest','False') == 'True':
                if self.numeric_feature:
                    index = cs.findiforestOutlier(self.data[self.numeric_feature])
                    inlier_indice = np.logical_and(inlier_indice, index)
                    self.log.info(f'-------> Numeric feature based Outlier detection(IsolationForest):')
        if inlier_indice.sum() != len(self.data):
            self.__update_index(inlier_indice, 'outlier')

    def fill_outlier_method(self,colm, method):
        if colm in self.numeric_feature:
            if method in cs.supported_method['outlier_column_wise']:
                if 'outlier' not in self.process_method.keys():
                    self.process_method['outlier'] = {}
                if method != 'Disable':
                    self.process_method['outlier'][colm] = method
            else:
                self.log.info(f"-------> outlier detection method '{method}' is not supported for column wise. supported methods are {cs.supported_method['outlier_column_wise']}")

    def fill_outlier_process(self,colm, method):
        if colm in self.numeric_feature:
            if method in cs.supported_method['outlierOperation']:
                if 'outlierOperation' not in self.process_method.keys():
                    self.process_method['outlierOperation'] = {}
                self.process_method['outlierOperation'][colm] = method
            else:
                self.log.info(f"-------> outlier process method '{method}' is not supported for column wise. supported methods are {cs.supported_method['outlierOperation']}")
        
    def get_cat_imputer(self,method):
        if method == 'mode':
            return SimpleImputer(strategy='most_frequent')
        elif method == 'zero':
            return SimpleImputer(strategy='constant', fill_value=0)
            
    def get_cat_encoder(self,method):
        if method == 'labelencoding':
            return OrdinalEncoder()
        elif method == 'onehotencoding':
            return OneHotEncoder(sparse=False,handle_unknown="ignore")
        elif method == 'targetencoding':
            if not self.is_target_available():
                raise ValueError('Can not apply Target Encoding when target feature is not present')
            return TargetEncoder()

    def get_num_imputer(self,method):
        if method == 'mode':
            return SimpleImputer(strategy='most_frequent')
        elif method == 'mean':
            return SimpleImputer(strategy='mean')
        elif method == 'median':
            return SimpleImputer(strategy='median')
        elif method == 'knnimputer':
            return KNNImputer()
        elif method == 'zero':
            return SimpleImputer(strategy='constant', fill_value=0)
            
    def get_num_scaler(self,method):
        if method == 'minmax':
            return MinMaxScaler()
        elif method == 'standardscaler':
            return StandardScaler()
        elif method == 'lognormal':
            return PowerTransformer(method='yeo-johnson', standardize=False)

    def recommenderStartProfiler(self,modelFeatures):
        return cs.recommenderStartProfiler(self,modelFeatures)
        
    def folderPreprocessing(self,folderlocation,folderdetails,deployLocation):
        return cs.folderPreprocessing(self,folderlocation,folderdetails,deployLocation)

    def textSimilarityStartProfiler(self, doc_col_1, doc_col_2):
        return cs.textSimilarityStartProfiler(self, doc_col_1, doc_col_2)

    def get_conversion_method(self):
        return cs.get_one_true_option(self.config.get('textConversionMethod','')).lower()

def set_features(features,profiler=None):
    return cs.set_features(features,profiler)
