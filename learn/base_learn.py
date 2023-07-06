import os
import math
import json
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from statistics import stdev

import sekitoba_library as lib
import sekitoba_data_manage as dm
from learn import data_adjustment

def lg_main( data ):
    params = {}
    
    if os.path.isfile( "best_params.json" ):
        f = open( "best_params.json", "r" )
        params = json.load( f )
        f.close()
    else:
        params["learning_rate"] = 0.01
        params["num_iteration"] = 10000
        params["max_depth"] = 200
        params["num_leaves"] = 175
        params["min_data_in_leaf"] = 25
        params["lambda_l1"] = 0
        params["lambda_l2"] = 0

    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ) )
    
    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': 'l2',
        'early_stopping_rounds': 30,
        'learning_rate': params["learning_rate"],
        'num_iteration': params["num_iteration"],
        'min_data_in_bin': 1,
        'max_depth': params["max_depth"],
        'num_leaves': params["num_leaves"],
        'min_data_in_leaf': params["min_data_in_leaf"],
        'lambda_l1': params["lambda_l1"],
        'lambda_l2': params["lambda_l2"]
    }

    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )
    
    dm.pickle_upload( lib.name.model_name(), bst )
        
    return bst
    
def importance_check( model ):
    result = []
    importance_data = model.feature_importance()
    f = open( "common/rank_score_data.txt" )
    all_data = f.readlines()
    f.close()
    c = 0

    for i in range( 0, len( all_data ) ):
        str_data = all_data[i].replace( "\n", "" )

        if "False" in str_data:
            continue

        result.append( { "key": str_data, "score": importance_data[c] } )
        c += 1

    result = sorted( result, key = lambda x: x["score"], reverse= True )

    wf = open( "importance_data.txt", "w" )

    for i in range( 0, len( result ) ):
        wf.write( "{}: {}\n".format( result[i]["key"], result[i]["score"] ) )        

def main( data, simu_data, learn = True ):
    learn_data = data_adjustment.data_check( data )

    if learn:
        model = lg_main( learn_data )
        importance_check( model )
    else:
        model = dm.pickle_load( lib.name.model_name() )
        
    data_adjustment.score_check( simu_data, model, upload = True )
    
    return model
