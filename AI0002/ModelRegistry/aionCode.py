#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This file is automatically generated by AION for AI0002_1 usecase.
File generation time: 2023-04-13 14:41:27
'''
#Standard Library modules
import sys
import json
import time
import platform
import tempfile
import sqlite3
import os
import logging
import shutil
import argparse
import warnings

#Third Party modules
import mlflow
from pathlib import Path
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

#local modules
from utility import *

warnings.filterwarnings("ignore")

IOFiles = {
    "log": "aion.log",
    "metaData": "modelMetaData.json",
    "model": "model.pkl",
    "performance": "performance.json",
    "production": "production.json",
    "monitor": "monitoring.json"
}

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

        
def validateConfig():        
    config_file = Path(__file__).parent/'config.json'        
    if not Path(config_file).exists():        
        raise ValueError(f'Config file is missing: {config_file}')        
    config = read_json(config_file)		
    return config		
        
class mlflow_register():        
        
    def __init__(self, input_path, model_name, meta_data):        
        self.input_path = Path(input_path).resolve()        
        self.model_name = model_name        
        self.meta_data = meta_data        
        self.logger = logging.getLogger('ModelRegister')        
        self.client = None        
        self.monitoring_data = read_json(self.input_path/IOFiles['monitor'])        
        mlflow_default_config = {'artifacts_uri':'','tracking_uri_type':'','tracking_uri':'','registry_uri':''}        
        if not self.monitoring_data.get('mlflow_config',False):        
            self.monitoring_data['mlflow_config'] = mlflow_default_config        
        
    def setup_registration(self):        
        tracking_uri, artifact_uri, registry_uri = get_mlflow_uris(self.monitoring_data['mlflow_config'],self.input_path)        
        self.logger.info(f'MLflow tracking uri: {tracking_uri}')        
        self.logger.info(f'MLflow registry uri: {registry_uri}')        
        mlflow.set_tracking_uri(tracking_uri)        
        mlflow.set_registry_uri(registry_uri)        
        self.client = mlflow.tracking.MlflowClient(        
                tracking_uri=tracking_uri,        
                registry_uri=registry_uri,        
            )        
        self.experiment_id = self.client.get_experiment_by_name(self.model_name).experiment_id        

    def __get_unprocessed_runs_sorted(self):
        query = "tags.processed = 'no'"
        runs = self.client.search_runs(
          experiment_ids=self.experiment_id,
          filter_string=query,
          run_view_type=ViewType.ACTIVE_ONLY,
          order_by=['metrics.test_score DESC']
        )
        return runs
        
    def __log_unprocessed_runs(self, runs):        
        self.logger.info('Unprocessed runs:')        
        for run in runs:        
            self.logger.info('	{}: {}'.format(run.info.run_id,run.data.metrics['test_score']))        
        
    def get_unprocessed_runs(self, model_path):        
        unprocessed_runs = self.__get_unprocessed_runs_sorted()        
        if not unprocessed_runs:        
            raise ValueError('Registering fail: No new trained model')        
        self.__log_unprocessed_runs( unprocessed_runs)        
        return unprocessed_runs        
        
    def __wait_until_ready(self, model_name, model_version):        
        client = MlflowClient()        
        for _ in range(10):        
            model_version_details = self.client.get_model_version(        
              name=model_name,        
              version=model_version,        
            )        
            status = ModelVersionStatus.from_string(model_version_details.status)        
            if status == ModelVersionStatus.READY:        
              break        
            time.sleep(1)        
        
    def __create_model(self, run):        
        artifact_path = 'model'        
        model_uri = 'runs:/{run_id}/{artifact_path}'.format(run_id=run.info.run_id, artifact_path=artifact_path)        
        self.logger.info(f'Registering model (run id): {run.info.run_id}')        
        model_details = mlflow.register_model(model_uri=model_uri, name=self.model_name)        
        self.__wait_until_ready(model_details.name, model_details.version)        
        self.client.set_tag(run.info.run_id, 'registered', 'yes' )        
        state_transition = self.client.transition_model_version_stage(        
            name=model_details.name,        
            version=model_details.version,        
            stage='Production',        
        )        
        self.logger.info(state_transition)        
        return model_details        
        
    def get_best_run(self, models):        
        return models[0]        
        
    def __validate_config(self):        
        try:        
            load_data_loc = self.meta_data['load_data']['Status']['DataFilePath']        
        except KeyError:        
            raise ValueError('DataIngestion step output is corrupted')        
        
    def __mlflow_log_transformer_steps(self, best_run):        
        run_id = best_run.info.run_id        
        meta_data = read_json(self.input_path/(best_run.data.tags['mlflow.runName']+'_'+IOFiles['metaData']))        
        self.__validate_config()        
        with mlflow.start_run(run_id):        
            if 'transformation' in meta_data.keys():        
                if 'target_encoder' in meta_data['transformation'].keys():        
                    source_loc = meta_data['transformation']['target_encoder']        
                    mlflow.log_artifact(str(self.input_path/source_loc))        
                    meta_data['transformation']['target_encoder'] = Path(source_loc).name        
                if 'preprocessor' in meta_data['transformation'].keys():        
                    source_loc = meta_data['transformation']['preprocessor']        
                    mlflow.log_artifact(str(self.input_path/source_loc))        
                    meta_data['transformation']['preprocessor'] = Path(source_loc).name        
        
            write_json(meta_data, self.input_path/IOFiles['metaData'])        
            mlflow.log_artifact(str(self.input_path/IOFiles['metaData']))        
        
    def __update_processing_tag(self, processed_runs):        
        self.logger.info('Changing status to processed:')        
        for run in processed_runs:        
            self.client.set_tag(run.info.run_id, 'processed', 'yes')        
            self.logger.info(f'	run id: {run.info.run_id}')        
        
    def update_unprocessed(self, runs):        
        return self.__update_processing_tag( runs)        
        
    def __force_register(self, best_run):        
        self.__create_model( best_run)        
        self.__mlflow_log_transformer_steps( best_run)        
        production_json = self.input_path/IOFiles['production']        
        production_model = {'Model':best_run.data.tags['mlflow.runName'],'runNo':self.monitoring_data['runNo'],'score':best_run.data.metrics['test_score']}        
        write_json(production_model, production_json)        
        database_path = self.input_path/(self.input_path.stem + '.db')        
        if database_path.exists():        
            database_path.unlink()        
        return best_run.data.tags['mlflow.runName']        
        
    def __get_register_model_score(self):        
        reg = self.client.list_registered_models()        
        if not reg:        
            return '', 0        
        run_id = reg[0].latest_versions[0].run_id        
        run = self.client.get_run(run_id)        
        score = run.data.metrics['test_score']        
        return run_id, score        
        
    def register_model(self, models, best_run):        
        return self.__force_register(best_run)
        
def __merge_logs(log_file_sequence,path, files):        
    if log_file_sequence['first'] in files:        
        with open(path/log_file_sequence['first'], 'r') as f:        
            main_log = f.read()        
        files.remove(log_file_sequence['first'])        
        for file in files:        
            with open(path/file, 'r') as f:        
                main_log = main_log + f.read()        
            (path/file).unlink()        
        with open(path/log_file_sequence['merged'], 'w') as f:        
            f.write(main_log)        
        
def merge_log_files(folder, models):        
    log_file_sequence = {        
        'first': 'aion.log',        
        'merged': 'aion.log'        
    }        
    log_file_suffix = '_aion.log'        
    log_files = [x+log_file_suffix for x in models if (folder/(x+log_file_suffix)).exists()]        
    log_files.append(log_file_sequence['first'])        
    __merge_logs(log_file_sequence, folder, log_files)        
        
def register_model(targetPath,models,usecasename, meta_data):        
    register = mlflow_register(targetPath, usecasename, meta_data)        
    register.setup_registration()        
        
    runs_with_score = register.get_unprocessed_runs(models)        
    best_run = register.get_best_run(runs_with_score)        
    register.update_unprocessed(runs_with_score)        
    return register.register_model(models, best_run)        
        
def register(log):        
    config = validateConfig()        
    targetPath = Path('aion')/config['targetPath']        
    models = config['models']        
    merge_log_files(targetPath, models)        
    meta_data_file = targetPath/IOFiles['metaData']        
    if meta_data_file.exists():        
        meta_data = read_json(meta_data_file)        
    else:        
        raise ValueError(f'Configuration file not found: {meta_data_file}')        
    usecase = config['targetPath']        
    # enable logging        
    log_file = targetPath/IOFiles['log']        
    log = logger(log_file, mode='a', logger_name=Path(__file__).parent.stem)        
    register_model_name = register_model(targetPath,models,usecase, meta_data)        
    status = {'Status':'Success','Message':f'Model Registered: {register_model_name}'}        
    log.info(f'output: {status}')        
    return json.dumps(status)
		
if __name__ == '__main__':        
    log = None        
    try:        
        print(register(log))        
    except Exception as e:        
        if log:        
            log.error(e, exc_info=True)        
        status = {'Status':'Failure','Message':str(e)}        
        print(json.dumps(status))