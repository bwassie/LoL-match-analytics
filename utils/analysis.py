import sys
sys.path.append(".//utils//")
import sqlite3, os
import pandas as pd
import parsing_utils
from parsing_utils import parse_databases
import importlib
importlib.reload(parsing_utils)
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
plt.style.use("ggplot")
from IPython.display import display_html
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import math
parser = parse_databases()
db_dir =  ".//databases"
match,random = parser.get_dbs(db_dir)
def drop_empty_cols(matches, minscale = .2):
    thresh = int(matches.shape[0]*.15)
    matches = matches[matches["league"]!= "LPL"].dropna(axis=1, thresh=thresh)
    matches["time"] = MinMaxScaler((minscale,1)).fit_transform(matches.loc[:,["date"]])
    return matches
query = """SELECT * FROM '2016matchdata';"""
matches_16 = drop_empty_cols(pd.read_sql_query(query,match))
query = """SELECT * FROM '2017matchdata';"""
matches_17 = drop_empty_cols(pd.read_sql_query(query,match))
query = """SELECT * FROM '2018matchdata';"""
matches_18 = drop_empty_cols(pd.read_sql_query(query,match), .4)

player_feats = ['gamelength', 'result', 'k', 'd', 'a', 'teamkills', 'teamdeaths', 'doubles', 'triples', 
                'quadras', 'pentas', 'fb', 'fbassist', 'fbvictim', 'fbtime', 'kpm', 'okpm', 'ckpm', 'fd', 
                'fdtime', 'teamdragkills', 'oppdragkills', 'herald', 'ft', 'fttime', 'firstmidouter', 
                'firsttothreetowers', 'teamtowerkills', 'opptowerkills', 'fbaron', 'fbarontime', 'teambaronkills', 
                'oppbaronkills', 'dmgtochamps', 'dmgtochampsperminute', 'dmgshare', 'earnedgoldshare', 'wards', 
                'wpm', 'wardshare', 'wardkills', 'wcpm', 'visionwards', 'visionwardbuys', 'visiblewardclearrate', 
                'invisiblewardclearrate', 'totalgold', 'earnedgpm', 'goldspent', 'gspd', 'minionkills', 
                'monsterkills', 'monsterkillsownjungle', 'monsterkillsenemyjungle', 'cspm', 'goldat10', 
                'oppgoldat10', 'gdat10', 'goldat15', 'oppgoldat15', 'gdat15', 'xpat10', 'oppxpat10', 'xpdat10']
				
all_feats, feats = make_feats(player_feats)
average = {}
past_matches = {2016:matches_16}#, 2017:matches_17}
past_data = {2016:players_2016, 2017:players_2017}
past_matches = {2016:matches_16, 2017:matches_17}
to_use = matches_18.copy()
match_data = pd.DataFrame(columns = all_feats+["league"], index = to_use["gameid"].unique()).fillna(0.0)
match_data["blue_win"] = 0
match_data, current_data = calculate_stats(to_use, past_data, past_matches, match_data)
#match_data.to_csv("2018_scale_test",sep="\t",index=False)

for ii in ["Red" , "Blue"]:
    side = {}
    for jj in ["Top", "Jungle", "Middle", "ADC", "Support"]:
        side[jj] =  get_average(past_matches, jj, ii, player_feats)
    average[ii] = side

def make_feats(player_feats, roles=["Top","Jungle","Middle","ADC","Support"]):
    blue_feats = {}
    red_feats = {}
    feats = {}
    all_feats = []
    if roles != None:
        for ii in roles:
            blue_feats[ii] = ["blue"+"_"+ii+"_"+x for x in player_feats]
            red_feats[ii] = ["red"+"_"+ii+"_"+x for x in player_feats]
            all_feats += blue_feats[ii]
            all_feats += red_feats[ii]
        feats = {"Red":red_feats,"Blue":blue_feats}
    else:
        blue_feats = ["blue"+"_"+x for x in player_feats]
        red_feats = ["red"+"_"+x for x in player_feats]
        all_feats += blue_feats
        all_feats += red_feats
        feats = {"Red":red_feats,"Blue":blue_feats}
    return all_feats, feats
    print(len(feats))

def get_player_dict(to_use, player_feats, scale = True):
    players_dict = pd.DataFrame(index=to_use["player"].unique(),columns=player_feats).fillna(0.0)
    players_dict["num_matches"] = 0
    group = to_use.groupby("player").groups
    for ii in group:
        player_stats = to_use[to_use["player"]==ii].apply(pd.to_numeric, errors = "coerce").fillna(0.0)
        #display(player_stats.loc[:,["k","time"]].mean(axis=0,skipna=False, numeric_only=None))
        if scale:
            players_dict.loc[ii,player_feats] = player_stats.loc[:,player_feats].multiply(player_stats["time"],axis="index").mean(axis=0,skipna=False, numeric_only=None)
        else:
            players_dict.loc[ii,player_feats] = player_stats.loc[:,player_feats].mean(axis=0,skipna=False, numeric_only=None)
        players_dict.loc[ii,"num_matches"] = player_stats.shape[0]
        #display(players_dict.loc[ii,"k"])
    return players_dict

def check_if_new(past_data, current_data, player):
    value = True
    for ii in past_data.keys():
        if player in past_data[ii].index.values:
            value = False
            return value
    for jj in current_data.keys():
        if player in jj:
            value=False
            return value
    return value

def get_average(past_data, role, side, player_feats):
    year = sorted(list(past_data.keys()),reverse=True)[0]
    subset = past_data[year].copy()
    subset.loc[:,player_feats] = subset.loc[:,player_feats].apply(pd.to_numeric, errors = "coerce").fillna(0.0)
    factor = 1
    average = factor*(subset[(subset["position"]==role) & (subset["side"] == side)].loc[:,player_feats]).mean(axis=0, skipna=False, numeric_only = None)
    return average.values

def update_current(current_data,game, player):
    if player not in current_data.keys():
        current_data[player] = game[game["player"] == player]
    else:
        current_data[player] = current_data[player].append(game[game["player"] == player])
    return current_data

def get_stats(past_data, current_data, player=None, player_feats=None):
    n = 0
    tmatches = 0
    stats = pd.Series(index = player_feats).fillna(0)
    if player in current_data.keys():
        #print("Current")
        n+=1
        data = current_data[player]
        stats = data.loc[:,player_feats].apply(pd.to_numeric, errors = "coerce").fillna(0.0).multiply(data["time"],axis="index").sum(axis=0, skipna=False, numeric_only = None)
        tmatches = data.shape[0]
    #print(stats.index.values)
    for ii in sorted(list(past_data.keys()),reverse=True):
        #print("Past")
        pdata = past_data[ii]
        scaling = 1.5**n
        n+=1
        try:
            stats += pdata.loc[player,"num_matches"]* pdata.loc[player,player_feats]/scaling
            tmatches += pdata.loc[player,"num_matches"]
        except:
            continue
    stats = stats/tmatches
    #print(stat_ids)
    return stats.values



def calculate_stats(to_use, past_data, past_matches, match_data):
    current_data = {}
    #display(match_data.head())
    n = 0
    new = 0
    times = []
    overall = time.time()
    for ii in to_use.groupby("gameid").groups:
        start = time.time()
        n+=1
        game = to_use[to_use["gameid"]==ii]

        if game[game["side"]=="Blue"]["result"].unique()[0] == 1:
            match_data.loc[ii,"blue_win"] = 1
        else:
            match_data.loc[ii,"blue_win"] = 0
        match_data.loc[ii,"league"] = game["league"].unique()[0]
        for _,jj in game.iterrows():
            if "team" not in jj["position"].lower():
                role = jj["position"]
                side = jj["side"]
                #print(game[game["player"]==jj]["position"].values)
                #print(game[game["player"]==jj]["side"].values)
                if check_if_new(past_data, current_data, jj["player"]):
                    new +=1
                    #average = get_average(past_matches, role, side, player_feats)
                    match_data.loc[ii,feats[side][role]] = average[side][role]
                else:
                    stats = get_stats(past_data, current_data,jj["player"], player_feats)
                    match_data.loc[ii,feats[side][role]] = stats
                current_data = update_current(current_data,game, jj["player"])
        delta = time.time() - start
        times.append(delta)
        if n % 100 == 0:
            diff = time.time() - overall
            print("Currently on match #: {}".format(n))
            print("Time elapsed to process last batch: {}".format(diff))
            overall = time.time()
        if n > 200000000:
            break
    #display(match_data.head())
    total = len(current_data.keys())
    print("# New Players: {}".format(new))
    print("# Total Players: {}".format(total))
    return match_data,  current_data

def split(match_data, split = True):
    data = match_data.dropna(how = "any", axis=0).drop("league",axis=1, errors = "ignore")
    data_init = pd.get_dummies(data)
    cols = data_init.drop("blue_win",axis=1).columns.values
    y = data_init["blue_win"].values
    scaler = StandardScaler().fit(data_init.drop("blue_win",axis=1))
    data = scaler.transform(data_init.drop("blue_win",axis=1))
    #print("Blue win rate: {:10.3f}".format(data_init["blue_win"].mean(axis=0)))
    if split:
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=21, shuffle=True, stratify = y)
        return y, cols, X_train, X_test, y_train, y_test, scaler
    else:
        return y, cols, data, data_init, scaler
		

def run_classifier(X_train, X_test, y_train, y_test, params = {"max_depth":1, "n_estimators":100, "learning_rate":.1, "colsample_bytree": .5, "subsample": .5, "gamma":0}, cv = True):
    cvresult = None
    clf = xgb.XGBClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], colsample_bytree = params["colsample_bytree"], subsample=params["subsample"], gamma = params["gamma"])
    xgb_param = clf.get_xgb_params()
    if cv:
        cvresult = cross_val_score(clf,X_train,y_train,cv = 5)
    clf.fit(X_train, y_train)
	
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print("Training accuracy: {:10.3f}".format(train_acc))
    
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print("Testing accuracy: {:10.3f}".format(test_acc))
    
    return clf, cvresult
