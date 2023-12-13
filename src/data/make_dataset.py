import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from openai import OpenAI
from datasets import load_dataset


class DataCleaner:
    
    def __init__(self) -> None:
        return
    
    def merge_dataset(self, json_files: list):
        glb_df = pd.DataFrame()

        if json_files:
            for json_file in json_files:
                df = pd.read_json(json_file)
                glb_df = pd.concat([glb_df, df])

            glb_df.to_csv("./data/processed/QuestionAnswers.csv", mode="a")                    

    def explore_dataset(self, json_file: str):
        data_df = pd.read_json(json_file)
        print("\n")
        print("\n")
        print(
            f"The dataset contains {data_df.shape[0]} rows and {data_df.shape[1]} columns"
        )
        print(
            "------------------------------------------------------------------------------"
        )
        print("\n")
        print(f"The columns names are: {data_df.columns}")
        print(
            "------------------------------------------------------------------------------"
        )
        print("\n")
        print(data_df.head())
        print(
            "------------------------------------------------------------------------------"
        )
        print("\n")
        print(data_df.info())
        print(
            "------------------------------------------------------------------------------"
        )
        print("\n")
        print(data_df.describe())
        print("\n")
        print("\n")

    def clean_dataset(
        self,
        json_file: str,
        method_to_remove_and_from: list = [],
        columns_to_be_removed: dict = {},
        col_values_to_be_added: dict = {},        
    ):
        df = pd.read_json(json_file)

        if method_to_remove_and_from:
            position = method_to_remove_and_from[0]["Method"][1]
            df.drop(df.iloc[:, position:], inplace=True, axis=1)

        if columns_to_be_removed:
            for col in columns_to_be_removed:
                try:
                    df.drop(col, inplace=True, axis=1)
                except Exception as error:
                    print("Error occured: ", error)

        if col_values_to_be_added:
            for col_value_pair in col_values_to_be_added:
                try:
                    df[col_value_pair] = col_values_to_be_added[col_value_pair]
                except Exception as error:
                    print("Error occured: ", error)

        # Update JSON file
        df.to_json(self.json_file)

    def rename_columns(self, json_file: str, columns_to_rename: dict):
        df = pd.read_csv(json_file)
        if columns_to_rename:
            for col in columns_to_rename:
                df.rename(
                    columns={
                        col: columns_to_rename[col],
                    },
                    inplace=True,
                )

        df.to_csv(json_file, index=False)


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

