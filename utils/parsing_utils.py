import sqlite3, os, glob
import pandas as pd
import numpy as np

class parse_databases():
    def get_dbs(self,db_dir):
        cdir = os.getcwd()
        os.chdir(db_dir)
        match = sqlite3.connect("match_data.db")
        match_cur = match.cursor()
        dbs = glob.glob("*.db")
        dbs.remove("match_data.db")
        aliases = []
        for ii in dbs:
            alias = ii.replace(".db","")
            aliases.append(alias)
            query = """ATTACH `{}` as {};""".format(ii,alias)
            cur_exec = match_cur.execute(query)
            #print("Attaching database {} to match database as {}".format(ii,alias))
        os.chdir(cdir)
        return match,aliases

    def get_tables(self,match, alias,joins, anti_joins=[]):
        tables = []
        join_str = ""
        anti_join_str = ""
        if len(joins) > 0:
            for ii in joins:
                join_str+="""AND name LIKE '%{}%'""".format(ii)
        if len(anti_joins) > 0:
            for ii in anti_joins:
                anti_join_str+="""AND name NOT LIKE '%{}%'""".format(ii)
        #print("\nExecuting SQL query:")
        query="""SELECT name FROM {}.'sqlite_master' WHERE type = "table" {} {};""".format(alias,join_str,anti_join_str)
        #print(query)
        cur_exec = match.execute(query)
        tables = [table[0] for table in cur_exec.fetchall()]
        return tables

    def create_table(self, db,name,alias,tables):
        match = db
        match_cur = match.cursor()
        try:
            query = """DROP TABLE "{}"; """.format(name)
            match_cur.execute(query)
            match.commit()
        except:
            print("Table {} does not exist, creating".format(name))
        teams = []
        for ii in tables:
            query = """SELECT 'table'.Team FROM {}.'{}' AS 'table'; """.format(alias, ii)
            df = pd.read_sql_query(query, match)
            for ii in df.iloc[:,0].values:
                teams.append(ii)
        teams = list(set(teams))
        df2 = pd.DataFrame({"Team":teams})
        df2.to_sql(name,match)
        return df2

    def get_matches_from_table(self, db,name,year='2017',league="",player="team"):
        match = db
        match_cur = match.cursor()
        query="""SELECT name FROM 'sqlite_master' WHERE type = "table" AND name LIKE "%match%" AND 
        name LIKE "%{}%";""".format(year)
        match_cur.execute(query)
        table = match_cur.fetchall()[0][0]
        league_str=""
        if league !="":
            league_str = """AND 'matches'.league LIKE '%{}%' """.format(league) 
        #print(tables)
        if player == "team":
            query = """SELECT * FROM '{}' AS 'matches' INNER JOIN '{}' AS 
                'table' ON 'table'.Team = 'matches'.team WHERE 'matches'.player LIKE 
                '%team%' {};""".format(table,name,league_str)
            matches = pd.read_sql_query(query, match).fillna(np.nan)
        else:
            query = """SELECT * FROM '{}' AS 'matches' INNER JOIN '{}' AS 
                'table' ON 'table'.Team = 'matches'.team WHERE 'matches'.player 
                NOT LIKE '%team%' {};""".format(table,name,league_str)
            matches = pd.read_sql_query(query, match).fillna(np.nan)
        return matches

    def parse_tables_get_matches(self,directory,alias,joins,anti_joins,name,year,league):
        match,aliases = self.get_dbs(directory)
        tables = self.get_tables(match,alias,joins,anti_joins)
        teams = self.create_table(match,name,alias, tables)
        matches = self.get_matches_from_table(match,name,year,league,joins[1])
        return matches, teams