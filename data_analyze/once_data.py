import math
import copy
from tqdm import tqdm
from mpi4py import MPI

import SekitobaLibrary as lib
import SekitobaDataManage as dm
import SekitobaPsql as ps

from SekitobaDataCreate.stride_ablity import StrideAblity
from SekitobaDataCreate.time_index_get import TimeIndexGet
from SekitobaDataCreate.jockey_data_get import JockeyAnalyze
from SekitobaDataCreate.trainer_data_get import TrainerAnalyze
from SekitobaDataCreate.high_level_data_get import RaceHighLevel
from SekitobaDataCreate.last_wrap import LastWrap
from SekitobaDataCreate.get_horce_data import GetHorceData

from common.name import Name

data_name = Name()

dm.dl.file_set( "predict_train_score.pickle" )
dm.dl.file_set( "predict_pace_data.pickle" )
dm.dl.file_set( "predict_last_passing_rank.pickle" )

class OnceData:
    def __init__( self ):
        self.race_data = ps.RaceData()
        self.race_horce_data = ps.RaceHorceData()
        self.horce_data = ps.HorceData()
        self.trainer_data = ps.TrainerData()
        self.jockey_data = ps.JockeyData()

        self.predict_pace_data = dm.dl.data_get( "predict_pace_data.pickle" )
        self.predict_last_passing_rank = dm.dl.data_get( "predict_last_passing_rank.pickle" )
        self.predict_train_score = dm.dl.data_get( "predict_train_score.pickle" )

        self.stride_ablity = StrideAblity( self.race_data )
        self.race_high_level = RaceHighLevel()
        self.time_index = TimeIndexGet( self.horce_data )
        self.trainer_analyze = TrainerAnalyze( self.race_data, self.race_horce_data, self.trainer_data )
        self.jockey_analyze = JockeyAnalyze( self.race_data, self.race_horce_data, self.jockey_data )
        self.last_wrap = LastWrap( self.race_data, self.horce_data, self.race_horce_data )

        self.data_name_list = []
        self.write_data_list = []
        self.simu_data = {}
        self.jockey_judgement_param_list = [ "limb", "popular", "flame_num", "dist", "kind", "baba", "place" ]
        self.trainer_judgement_param_list = [ "limb", "popular", "flame_num", "dist", "kind", "baba", "place" ]
        self.result = { "answer": [], "teacher": [], "query": [], "year": [], "level": [], "diff": [], "horce_body": [] }
        self.data_name_read()

    def data_name_read( self ):
        f = open( "common/list.txt", "r" )
        str_data_list = f.readlines()

        for str_data in str_data_list:
            self.data_name_list.append( str_data.replace( "\n", "" ) )

    def score_write( self ):
        f = open( "common/rank_score_data.txt", "w" )

        for data_name in self.write_data_list:
            f.write( data_name + "\n" )

        f.close()

    def data_list_create( self, data_dict ):
        result = []
        write_instance = []
        
        for data_name in self.data_name_list:
            try:
                result.append( data_dict[data_name] )
                write_instance.append( data_name )
            except:
                continue

        if len( self.write_data_list ) == 0:
            self.write_data_list = copy.deepcopy( write_instance )

        return result

    def division( self, score, d ):
        if score < 0:
            score *= -1
            score /= d
            score *= -1
        else:
            score /= d

        return int( score )

    def clear( self ):
        dm.dl.data_clear()
    
    def create( self, race_id ):
        self.race_data.get_all_data( race_id )
        self.race_horce_data.get_all_data( race_id )

        if len( self.race_horce_data.horce_id_list ) == 0:
            return

        self.horce_data.get_multi_data( self.race_horce_data.horce_id_list )
        self.trainer_data.get_multi_data( self.race_horce_data.trainer_id_list )
        self.jockey_data.get_multi_data( self.race_horce_data.jockey_id_list )

        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]
        key_place = str( self.race_data.data["place"] )
        key_dist = str( self.race_data.data["dist"] )
        key_kind = str( self.race_data.data["kind"] )      
        key_baba = str( self.race_data.data["baba"] )
        ymd = { "year": self.race_data.data["year"], \
               "month": self.race_data.data["month"], \
               "day": self.race_data.data["day"] }

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            return

        predict_pace = {}
        predict_netkeiba_pace = -1

        if race_id in self.predict_pace_data:
            predict_pace = self.predict_pace_data[race_id]

        predict_netkeiba_pace = lib.netkeibaPace( self.race_data.data["predict_netkeiba_pace"] )
        current_horce_body = self.race_data.data["corner_horce_body"]
        min_corner_key = lib.minCheck( self.race_data.data["corner_horce_body"] )
        key_race_money_class = str( int( lib.moneyClassGet( self.race_data.data["money"] ) ) )
        
        teacher_data = []
        answer_data = []
        answer_horce_body = []
        diff_data = []
        horce_id_list = []
        race_limb = {}
        getHorceDataDict: dict[ str, GetHorceData ] = {}
        current_race_data = {}
        current_race_data[data_name.my_limb_count] = { "-1": -1 }
        
        for name in self.data_name_list:
            if name in current_race_data:
                continue

            current_race_data[name] = []

        escape_limb1_count = 0
        escape_limb2_count = 0
        one_popular_limb = -1
        two_popular_limb = -1
        one_popular_odds = -1
        two_popular_odds = -1
        
        for horce_id in self.race_horce_data.horce_id_list:
            current_data, past_data = lib.raceCheck( self.horce_data.data[horce_id]["past_data"], ymd )
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data, self.race_data )

            if not cd.raceCheck():
                continue

            getHorceData = GetHorceData( cd, pd )
            getHorceDataDict[horce_id] = getHorceData

            if not getHorceData.limb_math == -1:
                lib.dicAppend( current_race_data[data_name.my_limb_count], getHorceData.key_limb, 0 )
                current_race_data[data_name.my_limb_count][getHorceData.key_limb] += 1
                
            race_limb[horce_id] = getHorceData.limb_math

            if getHorceData.limb_math == 1:
                escape_limb1_count += 1
            elif getHorceData.limb_math == 2:
                escape_limb2_count += 1

            escape_within_rank = -1
            
            if getHorceData.limb_math == 1 or getHorceData.limb_math == 2:
                escape_within_rank = cd.horceNumber()

            jockey_id = self.race_horce_data.data[horce_id]["jockey_id"]
            trainer_id = self.race_horce_data.data[horce_id]["trainer_id"]

            horce_true_skill = self.race_horce_data.data[horce_id]["horce_true_skill"]
            jockey_true_skill = self.race_horce_data.data[horce_id]["jockey_true_skill"]
            trainer_true_skill = self.race_horce_data.data[horce_id]["trainer_true_skill"]

            horce_first_passing_true_skill  =self.race_horce_data.data[horce_id]["horce_first_passing_true_skill"]
            jockey_first_passing_true_skill = self.race_horce_data.data[horce_id]["jockey_first_passing_true_skill"]
            trainer_first_passing_true_skill = self.race_horce_data.data[horce_id]["trainer_first_passing_true_skill"]

            horce_last_passing_true_skill  =self.race_horce_data.data[horce_id]["horce_last_passing_true_skill"]
            jockey_last_passing_true_skill = self.race_horce_data.data[horce_id]["jockey_last_passing_true_skill"]

            up3_horce_true_skill = self.race_horce_data.data[horce_id]["horce_up3_true_skill"]
            up3_jockey_true_skill = self.race_horce_data.data[horce_id]["jockey_up3_true_skill"]
            up3_trainer_true_skill = self.race_horce_data.data[horce_id]["trainer_up3_true_skill"]

            past_min_first_horce_body, past_max_first_horce_body, past_ave_first_horce_body, past_std_first_horce_body = \
              getHorceData.getFirstHorceBody()
            past_min_last_horce_body, past_max_last_horce_body, past_ave_last_horce_body, past_std_last_horce_body = \
              getHorceData.getLastHorceBody()

            popular = cd.popular()
            odds = cd.odds()

            if popular == 1:
                one_popular_limb = getHorceData.limb_math
                one_popular_odds = odds
            elif popular == 2:
                two_popular_limb = getHorceData.limb_math
                two_popular_odds = odds

            judgement_data = {}

            for param in self.jockey_judgement_param_list:
                try:
                    judgement_data["jockey_judgment_up3_{}".format( param )] = \
                      self.race_horce_data.data[horce_id]["jockey_judgment_up3"][param]
                except:
                    judgement_data["jockey_judgment_up3_{}".format( param )] = lib.escapeValue

            current_time_index = self.time_index.main( horce_id, pd.pastDayList() )
            speed, up_speed, pace_speed = pd.speedIndex( self.horce_data.data[horce_id]["baba_index"] )
            corner_diff_rank_ave = pd.corner_diff_rank()
            stride_ablity_data = self.stride_ablity.ablity_create( cd, pd )
            pace_up_rate = pd.pace_up_rate()

            for stride_data_key in stride_ablity_data.keys():
                current_race_data[stride_data_key].append( stride_ablity_data[stride_data_key] )

            current_race_data[data_name.horce_true_skill].append( horce_true_skill )
            current_race_data[data_name.jockey_true_skill].append( jockey_true_skill )
            current_race_data[data_name.trainer_true_skill].append( trainer_true_skill )
            current_race_data[data_name.horce_first_passing_true_skill].append( horce_first_passing_true_skill )
            current_race_data[data_name.jockey_first_passing_true_skill].append( jockey_first_passing_true_skill )
            current_race_data[data_name.trainer_first_passing_true_skill].append( trainer_first_passing_true_skill )
            current_race_data[data_name.horce_last_passing_true_skill].append( horce_last_passing_true_skill )
            current_race_data[data_name.jockey_last_passing_true_skill].append( jockey_last_passing_true_skill )
            current_race_data[data_name.up3_horce_true_skill].append( up3_horce_true_skill )
            current_race_data[data_name.up3_jockey_true_skill].append( up3_jockey_true_skill )
            current_race_data[data_name.up3_trainer_true_skill].append( up3_trainer_true_skill )
            current_race_data[data_name.corner_diff_rank_ave].append( corner_diff_rank_ave )
            current_race_data[data_name.speed_index].append( max( lib.maxCheck( speed ) + current_time_index["max"], \
                                                                 lib.escapeValue ) )
            current_race_data[data_name.up_rate].append( pd.up_rate( key_race_money_class, self.race_data.data["up_kind_ave"] ) )
            current_race_data[data_name.match_up3].append( pd.matchUp3() )
            current_race_data[data_name.max_up3].append( pd.maxUp3() )
            current_race_data[data_name.max_time_point].append( pd.maxTimePoint( self.race_data.data["race_time_analyze"] ) )
            current_race_data[data_name.max_up3_time_point].append( pd.maxUp3TimePoint( getHorceData.key_limb ) )
            current_race_data[data_name.min_up3].append( pd.minUp3() )
            current_race_data[data_name.burden_weight].append( cd.burdenWeight() )
            current_race_data[data_name.level_score].append( pd.level_score( self.race_data.data["money_class_true_skill"] ) )
            current_race_data[data_name.level_up3].append( pd.level_up3( self.race_data.data["money_class_true_skill"] ) )
            current_race_data[data_name.escape_within_rank].append( escape_within_rank )
            current_race_data[data_name.past_min_first_horce_body].append( past_min_first_horce_body )
            current_race_data[data_name.past_min_last_horce_body].append( past_min_last_horce_body )
            current_race_data[data_name.past_max_first_horce_body].append( past_max_first_horce_body )
            current_race_data[data_name.past_max_last_horce_body].append( past_max_last_horce_body )
            current_race_data[data_name.past_ave_first_horce_body].append( past_ave_first_horce_body )
            current_race_data[data_name.past_ave_last_horce_body].append( past_ave_last_horce_body )
            current_race_data[data_name.past_std_first_horce_body].append( past_std_first_horce_body )
            current_race_data[data_name.past_std_last_horce_body].append( past_std_last_horce_body )
            current_race_data[data_name.up_index].append( lib.maxCheck( up_speed ) )
            current_race_data[data_name.stamina].append( pd.stamina_create( getHorceData.key_limb ) )
            horce_id_list.append( horce_id )

            for judge_key in judgement_data.keys():
                lib.dicAppend( current_race_data, judge_key, [] )
                current_race_data[judge_key].append( judgement_data[judge_key] )

            for pace_up_rate_key in pace_up_rate.keys():
                current_race_data[data_name.pace_up_rate+"_"+pace_up_rate_key].append( pace_up_rate[pace_up_rate_key] )

        if len( horce_id_list ) < 2:
            return

        self.last_wrap.create_score()
        sort_race_data: dict[ str, list ] = {}
        ave_burden_weight = lib.average( current_race_data[data_name.burden_weight] )
        current_key_list = []
        current_race_data[data_name.escape_within_rank] = sorted( current_race_data[data_name.escape_within_rank], reverse = True )

        for data_key in current_race_data.keys():
            if not type( current_race_data[data_key] ) is list or \
              len( current_race_data[data_key] ) == 0:
                continue

            current_key_list.append( data_key )

        for data_key in current_key_list:
            current_race_data[data_key+"_index"] = sorted( current_race_data[data_key], reverse = True )
            current_race_data[data_key+"_stand"] = lib.standardization( current_race_data[data_key] )
            current_race_data[data_key+"_devi"] = lib.deviationValue( current_race_data[data_key] )

        N = len( horce_id_list )
        std_past_ave_first_horce_body = lib.stdev( current_race_data[data_name.past_ave_first_horce_body] )
        std_past_ave_last_horce_body = lib.stdev( current_race_data[data_name.past_ave_last_horce_body] )

        min_race_horce_true_skill = lib.minimum( current_race_data[data_name.horce_true_skill] )
        min_race_jockey_true_skill = lib.minimum( current_race_data[data_name.jockey_true_skill] )
        min_race_trainer_true_skill = lib.minimum( current_race_data[data_name.trainer_true_skill] )
        min_race_horce_first_passing_true_skill = lib.minimum( current_race_data[data_name.horce_first_passing_true_skill] )
        min_race_jockey_first_passing_true_skill = lib.minimum( current_race_data[data_name.jockey_first_passing_true_skill] )
        min_race_trainer_first_passing_true_skill = lib.minimum( current_race_data[data_name.trainer_first_passing_true_skill] )

        min_speed_index = lib.minimum( current_race_data[data_name.speed_index] )
        min_up_rate = lib.minimum( current_race_data[data_name.up_rate] )
        min_past_ave_first_horce_body = lib.minimum( current_race_data[data_name.past_ave_first_horce_body] )
        min_past_ave_last_horce_body = lib.minimum( current_race_data[data_name.past_ave_last_horce_body] )
        min_past_max_first_horce_body = lib.minimum( current_race_data[data_name.past_max_first_horce_body] )
        min_past_max_last_horce_body = lib.minimum( current_race_data[data_name.past_max_last_horce_body] )
        min_past_min_first_horce_body = lib.minimum( current_race_data[data_name.past_min_first_horce_body] )
        min_past_min_last_horce_body = lib.minimum( current_race_data[data_name.past_min_last_horce_body] )

        max_race_horce_true_skill = max( current_race_data[data_name.horce_true_skill] )
        max_race_jockey_true_skill = max( current_race_data[data_name.jockey_true_skill] )
        max_race_trainer_true_skill = max( current_race_data[data_name.trainer_true_skill] )
        max_race_horce_first_passing_true_skill = max( current_race_data[data_name.horce_first_passing_true_skill] )
        max_race_jockey_first_passing_true_skill = max( current_race_data[data_name.jockey_first_passing_true_skill] )
        max_race_trainer_first_passing_true_skill = max( current_race_data[data_name.trainer_first_passing_true_skill] )

        max_speed_index = max( current_race_data[data_name.speed_index] )
        max_up_rate = max( current_race_data[data_name.up_rate] )
        max_past_ave_first_horce_body = max( current_race_data[data_name.past_ave_first_horce_body] )
        max_past_ave_last_horce_body = max( current_race_data[data_name.past_ave_last_horce_body] )
        max_past_max_first_horce_body = max( current_race_data[data_name.past_max_first_horce_body] )
        max_past_max_last_horce_body = max( current_race_data[data_name.past_max_last_horce_body] )
        max_past_min_first_horce_body = max( current_race_data[data_name.past_min_first_horce_body] )
        max_past_min_last_horce_body = max( current_race_data[data_name.past_min_last_horce_body] )

        ave_race_horce_true_skill = lib.average( current_race_data[data_name.horce_true_skill] )
        ave_race_jockey_true_skill = lib.average( current_race_data[data_name.jockey_true_skill] )
        ave_race_trainer_true_skill = lib.average( current_race_data[data_name.trainer_true_skill] )
        ave_race_horce_first_passing_true_skill = lib.average( current_race_data[data_name.horce_first_passing_true_skill] )
        ave_race_jockey_first_passing_true_skill = lib.average( current_race_data[data_name.jockey_first_passing_true_skill] )
        ave_race_trainer_first_passing_true_skill = lib.average( current_race_data[data_name.trainer_first_passing_true_skill] )

        ave_speed_index = lib.average( current_race_data[data_name.speed_index] )
        ave_up_rate = lib.average( current_race_data[data_name.up_rate] )
        ave_past_ave_first_horce_body = lib.average( current_race_data[data_name.past_ave_first_horce_body] )
        ave_past_ave_last_horce_body = lib.average( current_race_data[data_name.past_ave_last_horce_body] )
        ave_past_max_first_horce_body = lib.average( current_race_data[data_name.past_max_first_horce_body] )
        ave_past_max_last_horce_body = lib.average( current_race_data[data_name.past_max_last_horce_body] )
        ave_past_min_first_horce_body = lib.average( current_race_data[data_name.past_min_first_horce_body] )
        ave_past_min_last_horce_body = lib.average( current_race_data[data_name.past_min_last_horce_body] )

        for count, horce_id in enumerate( horce_id_list ):
            current_data, past_data = lib.raceCheck( self.horce_data.data[horce_id]["past_data"], ymd )
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data, self.race_data )

            if not cd.raceCheck():
                continue

            #getHorceData = getHorceDataDict[horce_id]
            getHorceData = GetHorceData( cd, pd )
            place_num = int( race_place_num )
            current_year = cd.year()
            horce_birth_day = int( horce_id[0:4] )
            age = current_year - horce_birth_day
            before_speed_score = getHorceData.getBeforeSpeed()
            before_diff_score = getHorceData.getBeforeDiff()
            before_id_weight_score = getHorceData.getBeforeIdWeight()
            before_popular = getHorceData.getBeforePopular()
            before_rank = getHorceData.getBeforeRank()
            up3_standard_value = getHorceData.getUp3StandardValue()
            diff_load_weight = getHorceData.getDiffLoadWeight()
            popular_rank = getHorceData.getPopularRank()
            before_first_passing_rank, before_last_passing_rank = getHorceData.getBeforePassingRank()
            limb_math = race_limb[horce_id]
            escape_within_rank = lib.escapeValue

            if getHorceData.limb_math == 1 or getHorceData.limb_math == 2:
                escape_within_rank = current_race_data[data_name.escape_within_rank].index( getHorceData.horce_num )
            
            judgement_data = {}

            for param in self.jockey_judgement_param_list:
                jockey_judgment_up3_rate = { "1": -1000, "2": -1000, "3": -1000 }

                try:
                    jockey_judgment_up3_rate = self.race_horce_data.data[horce_id]["jockey_judgment_up3_rate"][param]
                except:
                    pass

                for key_class in [ "0", "1", "2" ]:
                    try:
                        judgement_data["jockey_judgment_up3_rate_{}_{}".format( param, key_class )] = jockey_judgment_up3_rate[key_class]
                    except:
                        judgement_data["jockey_judgment_up3_rate_{}_{}".format( param, key_class )] = -1

            predict_last_passing_rank = lib.escapeValue
            predict_last_passing_rank_index = lib.escapeValue
            predict_last_passing_rank_stand = lib.escapeValue

            if race_id in self.predict_last_passing_rank and horce_id in self.predict_last_passing_rank[race_id]:
                predict_last_passing_rank = self.predict_last_passing_rank[race_id][horce_id]["score"]
                predict_last_passing_rank_index = self.predict_last_passing_rank[race_id][horce_id]["index"]
                predict_last_passing_rank_stand = self.predict_last_passing_rank[race_id][horce_id]["stand"]

            predict_netkeiba_deployment = lib.escapeValue

            for t in range( 0, len( self.race_data.data["predict_netkeiba_deployment"] ) ):
                if int( getHorceData.horce_num ) in self.race_data.data["predict_netkeiba_deployment"][t]:
                    predict_netkeiba_deployment = t
                    break

            t_instance = {}
            for wrap_key in self.last_wrap.horce_wrap_score[horce_id].keys():
                t_instance[wrap_key] = self.last_wrap.horce_wrap_score[horce_id][wrap_key]
            
            for judge_key in judgement_data.keys():
                t_instance[judge_key] = judgement_data[judge_key]
            
            t_instance[data_name.all_horce_num] = cd.allHorceNum()
            t_instance[data_name.ave_burden_weight_diff] = \
              ave_burden_weight - current_race_data[data_name.burden_weight][count]
            t_instance[data_name.ave_first_passing_rank] = pd.first_passing_rank()
            t_instance[data_name.baba] = cd.babaStatus()
            t_instance[data_name.before_diff] = before_diff_score
            t_instance[data_name.before_first_passing_rank] = before_first_passing_rank
            t_instance[data_name.before_id_weight] = before_id_weight_score
            t_instance[data_name.before_last_passing_rank] = before_last_passing_rank
            t_instance[data_name.before_rank] = before_rank
            t_instance[data_name.dist_kind] = cd.distKind()
            t_instance[data_name.dist_kind_count] = pd.distKindCount()
            t_instance[data_name.escape_limb1_count] = escape_limb1_count
            t_instance[data_name.escape_limb2_count] = escape_limb2_count
            t_instance[data_name.escape_within_rank] = escape_within_rank
            t_instance[data_name.horce_num] = getHorceData.horce_num
            t_instance[data_name.horce_sex] = self.horce_data.data[horce_id]["sex"]
            t_instance[data_name.limb] = limb_math
            t_instance[data_name.my_limb_count] = current_race_data[data_name.my_limb_count][getHorceData.key_limb]
            t_instance[data_name.odds] = cd.odds()
            t_instance[data_name.one_popular_limb] = one_popular_limb
            t_instance[data_name.one_popular_odds] = one_popular_odds
            t_instance[data_name.place] = place_num
            t_instance[data_name.std_past_ave_first_horce_body] = std_past_ave_first_horce_body
            t_instance[data_name.std_past_ave_last_horce_body] = std_past_ave_last_horce_body
            t_instance[data_name.two_popular_limb] = two_popular_limb
            t_instance[data_name.two_popular_odds] = two_popular_odds
            t_instance[data_name.up3_standard_value] = up3_standard_value
            t_instance[data_name.weight] = cd.weight() / 10
            t_instance[data_name.weather] = cd.weather()
            t_instance[data_name.diff_load_weight] = diff_load_weight
            t_instance[data_name.popular] = cd.popular()
            t_instance[data_name.ave_horce_true_skill] = \
              lib.minus( ave_race_horce_true_skill, current_race_data[data_name.horce_true_skill][count] )
            t_instance[data_name.ave_jockey_true_skill] = \
              lib.minus( ave_race_jockey_true_skill, current_race_data[data_name.jockey_true_skill][count] )
            t_instance[data_name.ave_trainer_true_skill] = \
              lib.minus( ave_race_trainer_true_skill, current_race_data[data_name.trainer_true_skill][count] )
            t_instance[data_name.ave_horce_first_passing_true_skill] = \
              lib.minus( ave_race_horce_first_passing_true_skill, current_race_data[data_name.horce_first_passing_true_skill][count] )
            t_instance[data_name.ave_jockey_first_passing_true_skill] = \
              lib.minus( ave_race_jockey_first_passing_true_skill, current_race_data[data_name.jockey_first_passing_true_skill][count] )
            t_instance[data_name.ave_trainer_first_passing_true_skill] = \
              lib.minus( ave_race_trainer_first_passing_true_skill, current_race_data[data_name.trainer_first_passing_true_skill][count] )
            t_instance[data_name.ave_speed_index] = \
              lib.minus( ave_speed_index, current_race_data[data_name.speed_index][count] )
            t_instance[data_name.ave_up_rate] = \
              lib.minus( ave_up_rate, current_race_data[data_name.up_rate][count] )
            t_instance[data_name.ave_past_ave_first_horce_body] = \
              lib.minus( ave_past_ave_first_horce_body, current_race_data[data_name.past_ave_first_horce_body][count] )
            t_instance[data_name.ave_past_ave_last_horce_body] = \
              lib.minus( ave_past_ave_last_horce_body, current_race_data[data_name.past_ave_last_horce_body][count] )
            t_instance[data_name.ave_past_max_first_horce_body] = \
              lib.minus( ave_past_max_first_horce_body, current_race_data[data_name.past_max_first_horce_body][count] )
            t_instance[data_name.ave_past_max_last_horce_body] = \
              lib.minus( ave_past_max_last_horce_body, current_race_data[data_name.past_max_last_horce_body][count] )
            t_instance[data_name.ave_past_min_first_horce_body] = \
              lib.minus( ave_past_min_first_horce_body, current_race_data[data_name.past_min_first_horce_body][count] )
            t_instance[data_name.ave_past_min_last_horce_body] = \
              lib.minus( ave_past_min_last_horce_body, current_race_data[data_name.past_min_last_horce_body][count] )
            t_instance[data_name.max_horce_true_skill] = \
              lib.minus( max_race_horce_true_skill, current_race_data[data_name.horce_true_skill][count] )
            t_instance[data_name.max_jockey_true_skill] = \
              lib.minus( max_race_jockey_true_skill, current_race_data[data_name.jockey_true_skill][count] )
            t_instance[data_name.max_trainer_true_skill] = \
              lib.minus( max_race_trainer_true_skill, current_race_data[data_name.trainer_true_skill][count] )
            t_instance[data_name.max_horce_first_passing_true_skill] = \
              lib.minus( max_race_horce_first_passing_true_skill, current_race_data[data_name.horce_first_passing_true_skill][count] )
            t_instance[data_name.max_jockey_first_passing_true_skill] = \
              lib.minus( max_race_jockey_first_passing_true_skill, current_race_data[data_name.jockey_first_passing_true_skill][count] )
            t_instance[data_name.max_trainer_first_passing_true_skill] = \
              lib.minus( max_race_trainer_first_passing_true_skill, current_race_data[data_name.trainer_first_passing_true_skill][count] )
            t_instance[data_name.max_speed_index] = \
              lib.minus( max_speed_index, current_race_data[data_name.speed_index][count] )
            t_instance[data_name.max_up_rate] = \
              lib.minus( max_up_rate, current_race_data[data_name.up_rate][count] )
            t_instance[data_name.max_past_ave_first_horce_body] = \
              lib.minus( max_past_ave_first_horce_body, current_race_data[data_name.past_ave_first_horce_body][count] )
            t_instance[data_name.max_past_ave_last_horce_body] = \
              lib.minus( max_past_ave_last_horce_body, current_race_data[data_name.past_ave_last_horce_body][count] )
            t_instance[data_name.max_past_max_first_horce_body] = \
              lib.minus( max_past_max_first_horce_body, current_race_data[data_name.past_max_first_horce_body][count] )
            t_instance[data_name.max_past_max_last_horce_body] = \
              lib.minus( max_past_max_last_horce_body, current_race_data[data_name.past_max_last_horce_body][count] )
            t_instance[data_name.max_past_min_first_horce_body] = \
              lib.minus( max_past_min_first_horce_body, current_race_data[data_name.past_min_first_horce_body][count] )
            t_instance[data_name.max_past_min_last_horce_body] = \
              lib.minus( max_past_min_last_horce_body, current_race_data[data_name.past_min_last_horce_body][count] )
            t_instance[data_name.min_horce_true_skill] = \
              lib.minus( min_race_horce_true_skill, current_race_data[data_name.horce_true_skill][count] )
            t_instance[data_name.min_jockey_true_skill] = \
              lib.minus( min_race_jockey_true_skill, current_race_data[data_name.jockey_true_skill][count] )
            t_instance[data_name.min_trainer_true_skill] = \
              lib.minus( min_race_trainer_true_skill, current_race_data[data_name.trainer_true_skill][count] )
            t_instance[data_name.min_horce_first_passing_true_skill] = \
              lib.minus( min_race_horce_first_passing_true_skill, current_race_data[data_name.horce_first_passing_true_skill][count] )
            t_instance[data_name.min_jockey_first_passing_true_skill] = \
              lib.minus( min_race_jockey_first_passing_true_skill, current_race_data[data_name.jockey_first_passing_true_skill][count] )
            t_instance[data_name.min_trainer_first_passing_true_skill] = \
              lib.minus( min_race_trainer_first_passing_true_skill, current_race_data[data_name.trainer_first_passing_true_skill][count] )
            t_instance[data_name.min_speed_index] = \
              lib.minus( min_speed_index, current_race_data[data_name.speed_index][count] )
            t_instance[data_name.min_up_rate] = \
              lib.minus( min_up_rate, current_race_data[data_name.up_rate][count] )
            t_instance[data_name.three_average] = pd.threeAverage()
            t_instance[data_name.three_difference] = pd.threeDifference()
            t_instance[data_name.one_rate] = pd.oneRate()
            t_instance[data_name.two_rate] = pd.twoRate()
            t_instance[data_name.three_rate] = pd.threeRate()
            t_instance[data_name.match_rank] = pd.matchRank()
            t_instance[data_name.passing_regression] = pd.passing_regression()
            t_instance[data_name.average_speed] = pd.average_speed()
            t_instance[data_name.best_first_passing_rank] = pd.best_first_passing_rank()
            t_instance[data_name.best_second_passing_rank] = pd.best_second_passing_rank()
            t_instance[data_name.best_weight] = pd.bestWeight()
            t_instance[data_name.before_continue_not_three_rank] = pd.before_continue_not_three_rank()
            t_instance[data_name.diff_pace_time] = pd.diff_pace_time()
            t_instance[data_name.diff_pace_first_passing] = pd.diff_pace_first_passing()
            t_instance[data_name.pace_up] = pd.pace_up_check( self.race_data.data["up_pace_regressin"] )
            t_instance[data_name.high_level_score] = self.race_high_level.data_get( cd, pd, ymd )
            t_instance[data_name.jockey_rank] = self.jockey_analyze.rank( race_id, horce_id )
            t_instance[data_name.jockey_year_rank] = self.jockey_analyze.year_rank( horce_id, getHorceData.key_before_year )
            t_instance[data_name.trainer_rank] = self.trainer_analyze.rank( race_id, horce_id )
            t_instance[data_name.money] = pd.getMoney()
            t_instance[data_name.popular_rank] = popular_rank
            t_instance[data_name.before_speed] = before_speed_score
            t_instance[data_name.before_popular] = before_popular
            t_instance[data_name.predict_last_passing_rank] = predict_last_passing_rank
            t_instance[data_name.predict_last_passing_rank_index] = predict_last_passing_rank_index
            t_instance[data_name.predict_last_passing_rank_stand] = predict_last_passing_rank_stand
            t_instance[data_name.up_index] = current_race_data[data_name.up_index][count]
            t_instance[data_name.up_index_index] = \
              current_race_data[data_name.up_index_index].index( current_race_data[data_name.up_index][count] )
            t_instance[data_name.up_index_stand] = current_race_data[data_name.up_index_stand][count]
            t_instance[data_name.speed_index] = current_race_data[data_name.speed_index][count]
            t_instance[data_name.speed_index_index] = \
              current_race_data[data_name.speed_index_index].index( current_race_data[data_name.speed_index][count] )
            t_instance[data_name.speed_index_stand] = current_race_data[data_name.speed_index_stand][count]
            t_instance[data_name.predict_netkeiba_pace] = predict_netkeiba_pace
            t_instance[data_name.predict_netkeiba_deployment] = predict_netkeiba_deployment
            t_instance.update( getHorceData.getPredictPace( predict_pace ) )
            t_instance.update( lib.horceTeacherAnalyze( current_race_data, t_instance, count ) )

            t_list = self.data_list_create( t_instance )
            up3_time = cd.upTime()

            lib.dicAppend( self.simu_data, race_id, {} )
            self.simu_data[race_id][horce_id] = {}
            self.simu_data[race_id][horce_id]["data"] = t_list
            self.simu_data[race_id][horce_id]["answer"] = { "up3": up3_time,
                                                           "odds": cd.odds(),
                                                           "popular": cd.popular(),
                                                           "horce_num": getHorceData.horce_num }
            answer_data.append( up3_time )
            teacher_data.append( t_list )

        if not len( answer_data ) == 0:
            self.result["answer"].append( answer_data )
            self.result["teacher"].append( teacher_data )
            self.result["year"].append( year )
            self.result["query"].append( { "q": len( answer_data ), "year": year } )
