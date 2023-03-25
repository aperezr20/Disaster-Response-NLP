import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Returns a dataframe from merging the two dataframes loaded
    from the two filepaths

            Parameters:
                    messages_filepath (str): Path to the messages csv file
                    categories_filepath (str): Path to the categories csv file

            Returns:
                    df (DataFrame): DataFrame that merges messages and categories on id
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    merged_df = pd.merge(messages, categories, on="id")
    
    return merged_df
    

def clean_data(df):
    '''
    Cleans the dataframe by correctly splitting the categories column and removing duplicates

            Parameters:
                    df (DataFrame): Merged Dataframe

            Returns:
                    df (DataFrame): Clean DataFrame
    '''
    
    # Dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # First row of the categories dataframe
    row = categories.head(1).astype('str')
    
    # Get category column names by removing the last 3 characters 
    category_colnames = row.apply(lambda x: x.str[:-2], axis=1)
    
    # Set the categories column names to categories df
    categories.columns = category_colnames.loc[0]
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int) 
    
    # Drop the original categories column
    df = df.drop(['categories'],axis=1)
    # Concatenate the new set of categories columns
    df = pd.concat([df, categories], axis=1)
    # Drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    Saves the clean dataset into an sqlite database

            Parameters:
                    df (DataFrame): Clean DataFrame

            Returns:
                    None
    '''
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('table_disaster', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()