import math
import numpy as np

import SekitobaLibrary as lib
import SekitobaDataManage as dm

def data_check( data, state = "test" ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    result["query"] = []
    result["test_query"] = []

    for i in range( 0, len( data["teacher"] ) ):
        year = data["year"][i]
        query = len( data["teacher"][i] )
        data_check = lib.test_year_check( data["year"][i], state )

        if data_check == "test":
            result["test_query"].append( query )
        elif data_check == "teacher":
            result["query"].append( query )

        for r in range( 0, query ):
            current_data = data["teacher"][i][r]
            current_answer = data["answer"][i][r]

            if data_check == "test":
                result["test_teacher"].append( current_data )
                result["test_answer"].append( current_answer )
            elif data_check == "teacher":
                result["teacher"].append( current_data )
                result["answer"].append( current_answer  )

    return result

def score_check( simu_data, model, score_years = lib.test_years, upload = False ):
    score = 0
    count = 0
    simu_predict_data = {}
    predict_use_data = []

    for race_id in simu_data.keys():
        for horce_id in simu_data[race_id].keys():
            predict_use_data.append( simu_data[race_id][horce_id]["data"] )

    c = 0
    predict_data = model.predict( np.array( predict_use_data ) )

    for race_id in simu_data.keys():
        year = race_id[0:4]
        race_place_num = race_id[4:6]        
        check_data = []
        score_list = []
        simu_predict_data[race_id] = {}
        all_horce_num = len( simu_data[race_id] )
        
        for horce_id in simu_data[race_id].keys():
            predict_score = predict_data[c]
            answer_up3 = simu_data[race_id][horce_id]["answer"]["up3"]
            check_data.append( { "horce_id": horce_id, "answer": answer_up3, "score": predict_score } )
            score_list.append( predict_score )
            c += 1

        stand_score_list = lib.standardization( score_list )
        sort_score_list = sorted( score_list )
        check_data = sorted( check_data, key = lambda x: x["score"] )
        
        for i in range( 0, len( check_data ) ):
            predict_score = check_data[i]["score"]
            simu_predict_data[race_id][check_data[i]["horce_id"]] = {}
            simu_predict_data[race_id][check_data[i]["horce_id"]]["index"] = sort_score_list.index( predict_score )
            simu_predict_data[race_id][check_data[i]["horce_id"]]["score"] = predict_score
            simu_predict_data[race_id][check_data[i]["horce_id"]]["stand"] = stand_score_list[i]

            if year in score_years and not int( race_place_num ) == 8:
                score += math.pow( predict_score - check_data[i]["answer"], 2 )
                count += 1            
            
    score /= count
    score = math.sqrt( score )
    print( "score: {}".format( score ) )

    if upload:
        dm.pickle_upload( "predict_up3.pickle", simu_predict_data )

    return score
