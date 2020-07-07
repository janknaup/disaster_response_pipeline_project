import sys
import pandas as pd
import sqlite3


def get_categories(df: pd.DataFrame):
    """
    get category names from 1st item in data frame. assumes categories appear in same order in every row

    :param df: data frame to sample
    :return: list of category name strings
    """
    tokens = df.iloc[0]['categories'].split(";")
    return [tok.split("-")[0] for tok in tokens]


def get_category_values(line: str):
    """
    Parse a message categories line, return values as list

    :param line: categories string for single disaster message
    :return: list of 1-hot encoded categories
    """
    tokens = line.split(";")
    return pd.Series([int(tok.split("-")[1]) for tok in tokens])


def load_data(messages_filepath, categories_filepath):
    """
    loads messages and categories, joins into single data frame

    :param messages_filepath: path to the messages csv file
    :param categories_filepath: path to the categories csv file
    :return:
    """
    cats_raw_df = pd.read_csv(categories_filepath, header=0, index_col='id')
    messages = pd.read_csv(messages_filepath, header=0, index_col='id')
    return messages.join(cats_raw_df)


def clean_data(df):
    """
    splits categories string-blob column into 1-hot indicator columns
    removes messages with "NOTES:" marker, which are irrelevant or have individual problems

    :param df: input DataFrame
    :return: cleaned DataFrame
    """
    cat_columns = get_categories(df)
    cat_indicators = df['categories'].apply(get_category_values)
    cat_indicators.columns = cat_columns
    indicators_df = df.join(cat_indicators).drop('categories', axis=1)
    noted_indices = df.loc[df['message'].str.contains('NOTES:')].index
    cleaned = indicators_df.drop(noted_indices, axis=0)
    return cleaned


def save_data(df, database_filename):
    """
    Saves the disaster recovery data into table 'disaster_messages' the given sqlite3 database.
    Will overwrite an existing table.

    :param df: DataFrame with cleaned message and category data
    :param database_filename: sqlite3 file to write to
    :return:
    """
    with sqlite3.connect(database_filename) as conn:
        df.to_sql('disaster_messages', conn, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
