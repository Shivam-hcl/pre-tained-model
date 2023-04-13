#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This file is automatically generated by AION for AI0002_1 usecase.
File generation time: 2023-04-13 14:41:28
'''
#Standard Library modules
import sys
import math
import json
import shutil
import argparse
import platform

#Third Party modules
import scipy
import joblib
import mlflow
import sklearn
import numpy as np 
from pathlib import Path
import pandas as pd 
from xgboost import XGBClassifier

#local modules
from utility import *
from data_reader import *

IOFiles = {
    "inputData": "rawData.dat",
    "metaData": "modelMetaData.json",
    "performance": "performance.json",
    "monitor": "monitoring.json",
    "log": "predict.log",
    "prodData": "prodData"
}
output_file = { }

def get_mlflow_uris(config, path):                    
    artifact_uri = None                    
    tracking_uri_type = config.get('tracking_uri_type',None)                    
    if tracking_uri_type == 'localDB':                    
        tracking_uri = 'sqlite:///' + str(path.resolve()/'mlruns.db')                    
    elif tracking_uri_type == 'server' and config.get('tracking_uri', None):                    
        tracking_uri = config['tracking_uri']                    
        if config.get('artifacts_uri', None):                    
            if Path(config['artifacts_uri']).exists():                    
                artifact_uri = 'file:' + config['artifacts_uri']                    
            else:                    
                artifact_uri = config['artifacts_uri']                    
        else:                    
            artifact_uri = 'file:' + str(path.resolve()/'mlruns')                    
    else:                    
        tracking_uri = 'file:' + str(path.resolve()/'mlruns')                    
        artifact_uri = None                    
    if config.get('registry_uri', None):                    
        registry_uri = config['registry_uri']                    
    else:                    
        registry_uri = 'sqlite:///' + str(path.resolve()/'registry.db')                    
    return tracking_uri, artifact_uri, registry_uri                    



class deploy():

    def __init__(self, base_config, log=None):        
        self.targetPath = (Path('aion')/base_config['targetPath']).resolve()        
        if log:        
            self.logger = log        
        else:        
            log_file = self.targetPath/IOFiles['log']        
            self.logger = logger(log_file, mode='a', logger_name=Path(__file__).parent.stem)        
        try:        
            self.initialize(base_config)        
        except Exception as e:        
            self.logger.error(e, exc_info=True)        

    def initialize(self, base_config):        
        self.usecase = base_config['targetPath']        
        monitoring_data = read_json(self.targetPath/IOFiles['monitor'])        
        self.prod_db_type = monitoring_data['prod_db_type']        
        self.db_config = monitoring_data['db_config']        
        mlflow_default_config = {'artifacts_uri':'','tracking_uri_type':'','tracking_uri':'','registry_uri':''}        
        tracking_uri, artifact_uri, registry_uri = get_mlflow_uris(monitoring_data.get('mlflow_config',mlflow_default_config), self.targetPath)        
        mlflow.tracking.set_tracking_uri(tracking_uri)        
        mlflow.tracking.set_registry_uri(registry_uri)        
        client = mlflow.tracking.MlflowClient()        
        self.model_version = client.get_latest_versions(self.usecase, stages=['production'] )        
        model_version_uri = 'models:/{model_name}/production'.format(model_name=self.usecase)        
        self.model = mlflow.pyfunc.load_model(model_version_uri)        
        run = client.get_run(self.model.metadata.run_id)        
        if run.info.artifact_uri.startswith('file:'): #remove file:///        
            skip_name = 'file:'        
            if run.info.artifact_uri.startswith('file:///'):        
                skip_name = 'file:///'        
            self.artifact_path = Path(run.info.artifact_uri[len(skip_name) : ])        
            self.artifact_path_type = 'file'        
            meta_data = read_json(self.artifact_path/IOFiles['metaData'])        
        else:        
            self.artifact_path = run.info.artifact_uri        
            self.artifact_path_type = 'url'        
            meta_data_file = mlflow.artifacts.download_artifacts(self.artifact_path+'/'+IOFiles['metaData'])        
            meta_data = read_json(meta_data_file)        
        self.selected_features = meta_data['load_data']['selected_features']        
        self.train_features = meta_data['training']['features']            
        if self.artifact_path_type == 'url':            
            preprocessor_file = mlflow.artifacts.download_artifacts(self.artifact_path+'/'+meta_data['transformation']['preprocessor'])            
            target_encoder_file = mlflow.artifacts.download_artifacts(self.artifact_path+'/'+meta_data['transformation']['target_encoder'])            
        else:            
            preprocessor_file = self.artifact_path/meta_data['transformation']['preprocessor']            
            target_encoder_file = self.artifact_path/meta_data['transformation']['target_encoder']            
        self.target_encoder = joblib.load(target_encoder_file)            
        self.preprocessor = joblib.load(preprocessor_file)            
        self.preprocess_out_columns = meta_data['transformation']['preprocess_out_columns']            

    def write_to_db(self, data):
        prod_file = IOFiles['prodData']
        writer = dataReader(reader_type=self.prod_db_type,target_path=self.targetPath, config=self.db_config )
        writer.write(data, prod_file)
        writer.close()

    def predict(self, data=None):
        try:
            return self.__predict(data)
        except Exception as e:
            if self.logger:
                self.logger.error(e, exc_info=True)
            raise ValueError(json.dumps({'Status':'Failure', 'Message': str(e)}))

    def __predict(self, data=None):
        df = pd.DataFrame()
        jsonData = json.loads(data)
        df = pd.json_normalize(jsonData)
        if len(df) == 0:
            raise ValueError('No data record found')
        missing_features = [x for x in self.selected_features if x not in df.columns]
        if missing_features:
            raise ValueError(f'some feature/s is/are missing: {missing_features}')
        df_copy = df.copy()
        df = df[self.selected_features]
        df = self.preprocessor.transform(df)
        if isinstance(df, scipy.sparse.spmatrix):
            df = df.toarray()
        df = pd.DataFrame(df, columns=self.preprocess_out_columns)
        df = df[self.train_features]
        df = df.astype(np.float32)		
        output = pd.DataFrame(self.model._model_impl.predict_proba(df), columns=self.target_encoder.classes_)        
        df_copy['prediction'] = output.idxmax(axis=1)        
        self.write_to_db(df_copy)        
        df_copy['probability'] = output.max(axis=1).round(2)        
        df_copy['remarks'] = output.apply(lambda x: x.to_json(), axis=1)        
        output = df_copy.to_json(orient='records')
        return output