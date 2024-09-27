# data_processor.py

import pandas as pd
from config import CSV_FILE_PATH, TEXT_COLUMN

def load_data():
    """
    Load data from the CSV file and combine all columns into a text representation.
    """
    df = pd.read_csv(CSV_FILE_PATH)

    # Combine all columns into a 'text' column
    df['text'] = df.astype(str).apply(lambda x: ' | '.join(x), axis=1)

    texts = df[TEXT_COLUMN].fillna('').tolist()
    return texts
