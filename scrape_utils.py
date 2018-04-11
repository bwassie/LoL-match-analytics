import requests
import pandas as pd
from bs4 import BeautifulSoup
import sqlite3, os

class HTMLTableParser:

    def parse_url(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        try:
            return [(table['id'],self.parse_html_table(table)) for table in soup.find_all('table')]
        except:
            return [("table",self.parse_html_table(table)) for table in soup.find_all('table')]

    def parse_html_table(self, table):
        n_columns = 0
        n_rows=0
        column_names = []
        # Find number of rows and columns
        # we also find the column titles if we can
        for row in table.find_all('tr'):

            # Determine the number of rows in the table
            td_tags = row.find_all('td')
            if len(td_tags) > 0:
                n_rows+=1
                if n_columns == 0:
                    # Set the number of columns for our table
                    n_columns = len(td_tags)

            # Handle column names if we find them
            th_tags = row.find_all('th') 
            if len(th_tags) > 0 and len(column_names) == 0:
                for th in th_tags:
                    column_names.append(th.get_text())

        # Safeguard on Column Titles
        if len(column_names) > 0 and len(column_names) != n_columns:
            raise Exception("Column titles do not match the number of columns")

        columns = column_names if len(column_names) > 0 else range(0,n_columns)
        df = pd.DataFrame(columns = columns,
                          index= range(0,n_rows))
        row_marker = 0
        for row in table.find_all('tr'):
            column_marker = 0
            columns = row.find_all('td')
            for column in columns:
                df.iat[row_marker,column_marker] = column.get_text()
                column_marker += 1
            if len(columns) > 0:
                row_marker += 1
        # Convert to float if possible
        for col in df:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass
        return df

    def scrape_data(self,url,pattern, sql_name="test.db"):
        n=0
        try:
            conn = sqlite3.connect(sql_name)
            cursor=conn.cursor()
            print("Creating database {} and inserting tables...".format(sql_name))
        except:
            print("Database {} exists, moving on".format(sql_name))
        r = requests.get(url)
        soup = BeautifulSoup(r.text,"lxml")
        for link in soup.find_all("a"):
            link_str = link.get("href")
            if link_str != None and pattern in link_str:
                name = link_str.split("/")[-2]
                try:
                    n+=1
                    table = self.parse_url(link_str)[0][1]
                    table.to_sql(name,con=conn,if_exists='fail')
                    #print("Inserting table {} in databse {}".format(name,sql_name))
                except:
                    s=1
                    #print("Table {} exists, moving on".format(name))
        print("Inserted {} tables into {}".format(n,sql_name))