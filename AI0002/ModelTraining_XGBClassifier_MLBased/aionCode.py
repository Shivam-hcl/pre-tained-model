
#Standard Library modules
import warnings
import argparse
import importlib
import operator
import platform
import time
import sys
import json
import logging
import math

#Third Party modules
import joblib
import pandas as pd 
from pathlib import Path
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import mlflow
import sklearn

#local modules
from utility import *
warnings.filterwarnings("ignore")

model_name = 'XGBClassifier_MLBased'

IOFiles = {
    "inputData": "featureEngineeredData.dat",
    "testData": "test.dat",
    "metaData": "modelMetaData.json",
    "monitor": "monitoring.json",
    "log": "XGBClassifier_MLBased_aion.log",
    "model": "XGBClassifier_MLBased_model.pkl",
    "performance": "XGBClassifier_MLBased_performance.json",
    "metaDataOutput": "XGBClassifier_MLBased_modelMetaData.json"
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


def mlflow_create_experiment(config, path, name):                    
    tracking_uri, artifact_uri, registry_uri = get_mlflow_uris(config, path)                    
    mlflow.tracking.set_tracking_uri(tracking_uri)                    
    mlflow.tracking.set_registry_uri(registry_uri)                    
    client = mlflow.tracking.MlflowClient()                    
    experiment = client.get_experiment_by_name(name)                    
    if experiment:                    
        experiment_id = experiment.experiment_id                    
    else:                    
        experiment_id = client.create_experiment(name, artifact_uri)                    
    return client, experiment_id                    

                    
def scoring_criteria(score_param, problem_type, class_count):                    
    if problem_type == 'classification':                    
        scorer_mapping = {                    
                    'recall':{'binary_class': 'recall', 'multi_class': 'recall_weighted'},                    
                    'precision':{'binary_class': 'precision', 'multi_class': 'precision_weighted'},                    
                    'f1_score':{'binary_class': 'f1', 'multi_class': 'f1_weighted'},                    
                    'roc_auc':{'binary_class': 'roc_auc', 'multi_class': 'roc_auc_ovr_weighted'}                    
                   }                    
        if (score_param.lower() == 'roc_auc') and (class_count > 2):                    
            score_param = make_scorer(roc_auc_score, needs_proba=True,multi_class='ovr',average='weighted')                    
        else:                    
            class_type = 'binary_class' if class_count == 2 else 'multi_class'                    
            if score_param in scorer_mapping.keys():                    
                score_param = scorer_mapping[score_param][class_type]                    
            else:                    
                score_param = 'accuracy'                    
    return score_param

def mlflowSetPath(path, name):                    
    db_name = str(Path(path)/'mlruns')                    
    mlflow.set_tracking_uri('file:///' + db_name)                    
    mlflow.set_experiment(str(Path(path).name))                    


def logMlflow( params, metrices, estimator,tags={}, algoName=None):                    
    run_id = None                    
    for k,v in params.items():                    
        mlflow.log_param(k, v)                    
    for k,v in metrices.items():                    
        mlflow.log_metric(k, v)                    
    if 'CatBoost' in algoName:                    
        model_info = mlflow.catboost.log_model(estimator, 'model')                    
    else:                    
        model_info = mlflow.sklearn.log_model(sk_model=estimator, artifact_path='model')                    
    tags['processed'] = 'no'                    
    tags['registered'] = 'no'                    
    mlflow.set_tags(tags)                    
    if model_info:                    
        run_id = model_info.run_id                    
    return run_id                    

def get_classification_metrices( actual_values, predicted_values):                    
    result = {}                    
    accuracy_score = sklearn.metrics.accuracy_score(actual_values, predicted_values)                    
    avg_precision = sklearn.metrics.precision_score(actual_values, predicted_values,                    
        average='macro')                    
    avg_recall = sklearn.metrics.recall_score(actual_values, predicted_values,                    
        average='macro')                    
    avg_f1 = sklearn.metrics.f1_score(actual_values, predicted_values,                    
        average='macro')                    
                    
    result['accuracy'] = math.floor(accuracy_score*10000)/100                    
    result['precision'] = math.floor(avg_precision*10000)/100                    
    result['recall'] = math.floor(avg_recall*10000)/100                    
    result['f1'] = math.floor(avg_f1*10000)/100                    
    return result                    

        
def validateConfig():        
    config_file = Path(__file__).parent/'config.json'        
    if not Path(config_file).exists():        
        raise ValueError(f'Config file is missing: {config_file}')        
    config = read_json(config_file)        
    return config
        
def save_model( experiment_id, estimator, features, metrices, params,tags, scoring):        
        # mlflow log model, metrices and parameters        
        with mlflow.start_run(experiment_id = experiment_id, run_name = model_name):        
            return logMlflow(params, metrices, estimator, tags, model_name.split('_')[0])


def train(log):        
    config = validateConfig()        
    targetPath = Path('aion')/config['targetPath']        
    if not targetPath.exists():        
        raise ValueError(f'targetPath does not exist')        
    meta_data_file = targetPath/IOFiles['metaData']        
    if meta_data_file.exists():        
        meta_data = read_json(meta_data_file)        
    else:        
        raise ValueError(f'Configuration file not found: {meta_data_file}')        
    log_file = targetPath/IOFiles['log']        
    log = logger(log_file, mode='a', logger_name=Path(__file__).parent.stem)        
    dataLoc = targetPath/IOFiles['inputData']        
    if not dataLoc.exists():        
        return {'Status':'Failure','Message':'Data location does not exists.'}        
        
    status = dict()        
    usecase = config['targetPath']        
    df = pd.read_csv(dataLoc)        
    prev_step_output = meta_data['featureengineering']['Status']

    # split the data for training        
    selected_features = prev_step_output['selected_features']        
    target_feature = config['target_feature']        
    train_features = prev_step_output['total_features'].copy()        
    train_features.remove(target_feature)        
    X_train = df[train_features]        
    y_train = df[target_feature]        
    test_data = read_data(targetPath/IOFiles['testData'])        
    X_test = test_data[train_features]        
    y_test = test_data[target_feature]
    log.info('Data balancing done')
    
    #select scorer
    scorer = scoring_criteria(config['scoring_criteria'],config['problem_type'], df[target_feature].nunique())
    log.info('Scoring criteria: accuracy')
    
    #Training model
    log.info('Training XGBClassifier for modelBased')
    features = selected_features['modelBased']            
    estimator = XGBClassifier()            
    param = config['algorithms']['XGBClassifier']
    grid = RandomizedSearchCV(estimator, param, scoring=scorer, n_iter=config['optimization_param']['iterations'],cv=config['optimization_param']['trainTestCVSplit'])            
    grid.fit(X_train[features], y_train)            
    train_score = grid.best_score_ * 100            
    best_params = grid.best_params_            
    estimator = grid.best_estimator_
    
    #model evaluation
    y_pred = estimator.predict(X_test[features])
    test_score = round(accuracy_score(y_test,y_pred),2) * 100
    log.info('Confusion Matrix:')
    log.info('\n' + pd.DataFrame(confusion_matrix(y_test,y_pred)).to_string())
    metrices = get_classification_metrices(y_test,y_pred)
    metrices.update({'train_score': train_score, 'test_score':test_score})
        
    meta_data['training'] = {}        
    meta_data['training']['features'] = features        
    scoring = config['scoring_criteria']        
    tags = {'estimator_name': model_name}        
    monitoring_data = read_json(targetPath/IOFiles['monitor'])        
    mlflow_default_config = {'artifacts_uri':'','tracking_uri_type':'','tracking_uri':'','registry_uri':''}        
    mlflow_client, experiment_id = mlflow_create_experiment(monitoring_data.get('mlflow_config',mlflow_default_config), targetPath, usecase)        
    run_id = save_model(experiment_id, estimator,features, metrices,best_params,tags,scoring)        
    write_json(meta_data,  targetPath/IOFiles['metaDataOutput'])        
    write_json({'scoring_criteria': scoring, 'metrices':metrices, 'param':best_params},  targetPath/IOFiles['performance'])        
        
    # return status        
    status = {'Status':'Success','mlflow_run_id':run_id,'FeaturesUsed':features,'test_score':metrices['test_score'],'train_score':metrices['train_score']}        
    log.info(f'Test score: {test_score}')        
    log.info(f'Train score: {train_score}')        
    log.info(f'MLflow run id: {run_id}')        
    log.info(f'output: {status}')        
    return json.dumps(status)
        
if __name__ == '__main__':        
    log = None        
    try:        
        print(train(log))        
    except Exception as e:        
        if log:        
            log.error(e, exc_info=True)        
        status = {'Status':'Failure','Message':str(e)}        
        print(json.dumps(status))        