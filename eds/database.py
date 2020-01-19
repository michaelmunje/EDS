import pandas as pd
import sqlite3


class Database:
    def __init__(self, db_loc: str) -> sqlite3.Database:
        self.connection = sqlite3.connect(db_loc)
        self.cursor = self.connection.cursor()

    def execute_command(self, statement: str) -> None:
        '''
        Attempts to execute a command
        '''
        try:
            self.cursor.execute(statement)
        except sqlite3.Error as e:
            print(e)

    def fetch_table_names(self) -> [str]:
        '''
        Retrieves all tables in the database
        '''
        self.execute_command('SELECT name FROM sqlite_master WHERE type = \'table\';')
        return [x[0] for x in self.cursor.fetchall()]

    def fetch_column_names(self, table: str) -> [str]:
        '''
        Retrieves all columns in a table
        '''
        self.execute_command('PRAGMA table_info(' + table + ');')
        return [x[1] for x in self.cursor.fetchall()]

    def fetch_col_values(self, table: str, col: str) -> []:
        '''
        Returns all values for a specific column
        '''
        self.execute_command('SELECT ' + col + ' FROM ' + table + ';')
        return [x[0] for x in self.cursor.fetchall()]


def rename_bad_cols(df: pd.DataFrame, chs: [str] = ['\'', '-', ' ', '(', ')', '/']) -> pd.DataFrame:
    '''
    In the DataFrame column names, replaces characters in chs with _
    '''
    for ch in chs:  # Assumes columns names are not very similar, i.e. Feat1\ and Feat1-
        df.columns = df.columns.str.replace(ch, '_')
    return df.columns


def df_to_database(df: pd.DataFrame, db_loc: str, table_name: str) -> None:
    df.columns = rename_bad_cols(df)
    database = Database(db_loc)
    for column in df.columns:
        database.execute_command(get_column_insertion_command(table_name, column))
    df.to_sql(table_name, database.connection, if_exists='append', index=False)


def get_table_create_command(table: str) -> str:
    '''
    Generates SQL command string for table creation in SQL
    '''
    return ('CREATE TABLE IF NOT EXISTS ' + table + ' (rowid INTEGER PRIMARY KEY);')


def get_column_insertion_command(table: str, column: str) -> str:
    '''
    Generates SQL command string for column insertion in table
    '''
    return ('ALTER TABLE ' + table + ' ADD ' + column + ' VARCHAR;')
