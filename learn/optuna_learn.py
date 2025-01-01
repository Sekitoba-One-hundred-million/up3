import math
import json
import optuna
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from statistics import stdev

import SekitobaLibrary as lib
import SekitobaDataManage as dm
from learn import data_adjustment

data = {}
simu_data = {}

def objective( trial ):
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ) )

    learning_rate = trial.suggest_float( 'learning_rate', 0.01, 0.05 )
    num_leaves =  trial.suggest_int( "num_leaves", 50, 300 )
    max_depth = trial.suggest_int( "max_depth", 200, 500 )
    num_iteration = trial.suggest_int( "num_iteration", 5000, 15000 )
    min_data_in_leaf = trial.suggest_int( "min_data_in_leaf", 1, 50 )
    lambda_l1 = trial.suggest_float( "lambda_l1", 0, 0.1 )
    lambda_l2 = trial.suggest_float( "lambda_l2", 0, 0.1 )

    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'l2',
        'early_stopping_rounds': 30,
        'learning_rate': learning_rate,
        'num_iteration': num_iteration,
        'min_data_in_bin': 1,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2
    }

    model = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     num_boost_round = 5000 )

    score = data_adjustment.score_check( simu_data, [ model ], score_years = lib.score_years )
    
    return score * 100

def optuna_main( arg_data, arg_simu_data ):
    global data
    global simu_data
    simu_data = arg_simu_data
    data = data_adjustment.data_check( arg_data, state = "optuna" )
    paramList = []

    for i in range( 0, 5 ):
        study = optuna.create_study()
        study.optimize(objective, n_trials=3)
        paramList.append( study.best_params )
    
    f = open( "best_params.json", "w" )
    json.dump( paramList, f )
    f.close()
