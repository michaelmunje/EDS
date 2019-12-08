import pandas as pd
import sqlite3
import pandas as pd
import glob
from sqlite3 import Error


def db_execute_command(con: sqlite3.Connection, statement: str) -> None:
    '''
    Attempts to execute a command
    '''
    
    try:
        con.cursor()
        con.execute(statement)
    except Error as e:
        print(e)
        

def get_table_create_command(table: str) -> str:
    '''
    Inserts a string representing a new table into a template for table creation in SQL
    '''
    
    return ('CREATE TABLE IF NOT EXISTS ' + table + ' (rowid INTEGER PRIMARY KEY);')


def get_column_insertion_command(table: str, column: str) -> str:
    '''
    Inserts strings representing a table and new column
    into a template for column creation in SQL
    '''
    
    return ('ALTER TABLE ' + table + ' ADD ' + column + ' VARCHAR;')


def parse_df_to_database(con: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    '''
    Adds each column of a dataframe to the sql table
    Then inserts all dataframe entries to the table
    '''
    
    for column in df.columns:
        db_execute_command(con, get_column_insertion_command(table_name, column))
    df.to_sql(table_name, con, if_exists='append', index=False)


def replace_col_strings(df: pd.DataFrame, chs: [str] = ['\'', '-', ' ', '(', ')', '/']) -> pd.DataFrame:
    '''
    In the DataFrame column names, replaces characters in chs with _
    '''
    for ch in chs:
        df.columns = df.columns.str.replace(ch, '_')
    return df.columns


def fetch_table_names(cur: sqlite3.Cursor) -> [str]:
    '''
    Retrieves all tables in the database
    '''
    
    cur.execute('SELECT name FROM sqlite_master WHERE type = \'table\';')
    return [x[0] for x in cur.fetchall()]


def fetch_column_names(cur: sqlite3.Cursor, table: str) -> [str]:
    '''
    Retrieves all columns in a table
    '''
    
    cur.execute('PRAGMA table_info(' + table + ');')
    return [x[1] for x in cur.fetchall()]


def fetch_col_values(cur: sqlite3.Cursor, table: str, col: str) -> []:
    '''
    Returns all values for a specific column
    '''
    
    cur.execute('SELECT ' + col + ' FROM ' + table + ';')
    return [x[0] for x in cur.fetchall()]