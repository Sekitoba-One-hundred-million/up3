import os
import math
import json
import numpy as np
import lightgbm as lgb
from tqdm import tqdm
from statistics import stdev

import SekitobaLibrary as lib
import SekitobaDataManage as dm
from learn import data_adjustment

def lg_main( data, category_index_list, index = None ):
    params = {}
    
    if os.path.isfile( "best_params.json" ) and not index is None:
        f = open( "best_params.json", "r" )
        params = json.load( f )[index]
        f.close()
    else:
        params["learning_rate"] = 0.01
        params["num_iteration"] = 10000
        params["max_depth"] = 200
        params["num_leaves"] = 175
        params["min_data_in_leaf"] = 25
        params["lambda_l1"] = 0
        params["lambda_l2"] = 0

    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ),
                             np.array( data["test_answer"] ),
                             categorical_feature = category_index_list,
                             free_raw_data = False )
    
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
        'lambda_l2': params["lambda_l2"],
        'device_type': 'cuda'
    }

    n_splits = 5
    model_index_data = {}
    test_model_list = []
    n = int( len( data["race_id"] ) / 5 + 1 )

    for i in range( 0, n_splits ):
        s = int( n * i )
        e = min( int( n * ( i + 1 ) ), len( data["race_id"] ) )
        print( s, e )
        use_race_id_list = data["race_id"][:s] + data["race_id"][e:]
        use_teacher = data["teacher"][:s] + data["teacher"][e:]
        use_answer = data["answer"][:s] + data["answer"][e:]
        lgb_train = lgb.Dataset( np.array( use_teacher ),
                                 np.array( use_answer ),
                                 categorical_feature = category_index_list )
        test_model = lgb.train( params = lgbm_params,
                                train_set = lgb_train,     
                                valid_sets = [lgb_train, lgb_vaild],
                                num_boost_round = 5000 )
        test_model_list.append( test_model )

        for race_id in use_race_id_list:
            model_index_data[race_id] = i

    lgb_train = lgb.Dataset( np.array( data["teacher"] ),
                             np.array( data["answer"] ),
                             categorical_feature = category_index_list )
            
    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     num_boost_round = 5000 )

    return bst, test_model_list, model_index_data

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

def main( data, simu_data, state = "test" ):
    model_list = []
    test_model_index_list = []
    test_model_list = []
    category_index_list = lib.create_category_index( data["category"] )
    learn_data = data_adjustment.data_check( data, state = state )

    for i in range( 0, 5 ):
        model, test_model_data, model_index_data = lg_main( learn_data, category_index_list, index = i )
        model_list.append( model )
        test_model_list.append( test_model_data )
        test_model_index_list.append( model_index_data )
        
    importance_check( model_list[0] )
    data_adjustment.score_check( simu_data, model_list, test_model_list, test_model_index_list,
                                 score_years = lib.simu_years, upload = True )
    dm.pickle_upload( lib.name.model_name(), model_list )
