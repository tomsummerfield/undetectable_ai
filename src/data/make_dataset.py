import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from openai import OpenAI
from datasets import load_dataset
import string


class DataCleaner:

    def __init__(self) -> None:
        pass

    def clean_dataset(
        self,
        file_type: str,
        file: str,
        columns_to_be_removed: list = [],
        col_values_to_be_added: dict = {},
    ) -> pd.DataFrame() :
        
        df: pd.DataFrame() = pd.DataFrame()                        
        
        if file_type == "json":
            df = pd.read_json(file)
        elif file_type == "csv" or file_type == "text":
            df = pd.read_csv(file, index_col=0)
        else:
            print("Unknown file type")
            return
     
        if columns_to_be_removed:
            for col in columns_to_be_removed:
                try:
                    df.drop(col, inplace=True, axis=1, index=None)                    
                except Exception as error:
                    print("Error occured: ", error)

        if col_values_to_be_added:
            for col_value_pair in col_values_to_be_added:
                try:
                    df[col_value_pair] = col_values_to_be_added[col_value_pair]
                except Exception as error:
                    print("Error occured: ", error)

        df['Response'] = df['Response'].str.lower()
        df['Response'] = df['Response'].str.replace(f"[{string.punctuation}]", '', regex=True)
        df['Response'] = df['Response'].str.replace('\d+', '', regex=True)
        df['Response'] = df['Response'].str.replace('â€“', '', regex=False)
        
        return df

class DataManager:

    def __init__(self) -> None:
        pass

    def save_dataset(self, data: any, save_format: str, path: str) -> None: 
        
        df: any = pd.DataFrame(data)    
        
        if save_format == "json":
            df.to_json(path)
        elif save_format == "csv":
            df.to_csv(path)
        elif save_format == "text":
            df.to_csv(path)
        else:
            print("Unkown")

class DataSplitter:

    def __init__(self) -> None:
        pass

    def split_dataset(
        self, csv: str, features_column: list, prediction_columns: list
    ) -> any:
        df = pd.read_csv(csv)
        train_size = int(len(df) * 0.8)
        test_size = len(df) - train_size
        x_train = df[features_column][:train_size]
        y_train = df[prediction_columns][:train_size]
        x_test = df[features_column][test_size:]
        y_test = df[prediction_columns][test_size:]
        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

class GPTResponse:
    def __init__(self, client: any) -> None:
        self.client = client

    # Method to retrieve GPT responses
    def return_gpt_response(self, question):
        # Return chat completion passing in the question
        chat_completion = self.client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{"role": "user", "content": question}],
        )
        return chat_completion.choices[0].message.content
