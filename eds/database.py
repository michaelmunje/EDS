import pandas as pd
import sqlite3


class Database:
    def __init__(self, db_loc: str):
        """
        Wrapper interface for a sqlite database.

        Parameters
        ----------
        db_loc : str
            File location of the database.
        """
        self.connection = sqlite3.connect(db_loc)
        self.cursor = self.connection.cursor()

    def execute_command(self, statement: str) -> None:
        """
        Executes an SQL command for the sqlite database.

        Parameters
        ----------
        statement : str
            Proper SQL command
        """
        try:
            self.cursor.execute(statement)
        except sqlite3.Error as e:
            print(e)

    def fetch_table_names(self) -> [str]:
        """
        Fetches the table names of the database.

        Returns
        -------
        [str]
            A list of table names of the database.
        """
        self.execute_command('SELECT name FROM sqlite_master WHERE type = \'table\';')
        return [x[0] for x in self.cursor.fetchall()]

    def fetch_column_names(self, table: str) -> [str]:
        """
        Fetches the column names for a particular table in the database.

        Parameters
        ----------
        table : str
            Table name in the database

        Returns
        -------
        [str]
            A list of column names for the table queried.
        """
        self.execute_command('PRAGMA table_info(' + table + ');')
        return [x[1] for x in self.cursor.fetchall()]

    def fetch_column_values(self, table: str, column: str) -> []:
        """
        Fetches a column's values for a particular table in the database.

        Parameters
        ----------
        table : str
            Table name in the database

        column: str
            Column name for the table

        Returns
        -------
        []
            A list of values in the particular column.
        """
        self.execute_command('SELECT ' + column + ' FROM ' + table + ';')
        return [x[0] for x in self.cursor.fetchall()]


def rename_bad_cols(df: pd.DataFrame, chs: [str] = ['\'', '-', ' ', '(', ')', '/']) -> pd.DataFrame:
    """
    Renames the columns with bad names in a dataframe.
    'Bad names' as in nonacceptable for a database column name.
    Important Note: 'Feature1-' & 'Feature1)' lead to an ambiguity issue.
    This should be handled by mantaining a hashed dictionary of the new columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to convert bad column names from.

    chs, optional
        List of unacceptable characters, by default ['\'', '-', ' ', '(', ')', '/'].

    Returns
    -------
    pd.DataFrame
        DataFrame that has no unacceptable column names.
    """
    for ch in chs:
        df.columns = df.columns.str.replace(ch, '_')
    return df.columns


def df_to_database(df_in: pd.DataFrame, db_loc: str, table_name: str) -> None:
    """
    Converts a dataframe to a database.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing data.

    db_loc : str
        File location for output database.

    table_name : str
        Name of table to store particular DataFrame.

        >>> from sklearn.datasets import load_boston
        >>> import pandas as pd
        >>> boston = load_boston()
        >>> df = pd.DataFrame(boston.data, columns=boston.feature_names)
        >>> df_to_database(df, 'tmp_database.db', 'boston')
    """
    df = df_in.copy()
    df.columns = rename_bad_cols(df)
    database = Database(db_loc)
    database.execute_command(get_table_create_command(table_name))
    for column in df.columns:
        database.execute_command(get_column_insertion_command(table_name, column))
    df.to_sql(table_name, database.connection, if_exists='append', index=False)


def get_table_create_command(table: str) -> str:
    """
    Given a table name, generates SQL command to create the table.

    Parameters
    ----------
    table : str
        Table name.

    Returns
    -------
    str
        SQL command for creating the particular table.

    Example of generated query from a table string:
        >>> get_table_create_command('example_table')
        'CREATE TABLE IF NOT EXISTS example_table (rowid INTEGER PRIMARY KEY);'
    """
    return ('CREATE TABLE IF NOT EXISTS ' + table + ' (rowid INTEGER PRIMARY KEY);')


def get_column_insertion_command(table: str, column: str) -> str:
    """
    Given a column name, generates SQL command to create the column in a particular table.

    Parameters
    ----------
    table : str
        Table name to create column in.

    column: str
        Column name.

    Returns
    -------
    str
        SQL command for creating the particular column in the table.

    Example of generated query from a table string:
        >>> get_column_insertion_command('example_table', 'new_column')
        'ALTER TABLE example_table ADD new_column VARCHAR;'
    """
    return ('ALTER TABLE ' + table + ' ADD ' + column + ' VARCHAR;')
