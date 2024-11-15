import os
from datasets import load_dataset, Dataset
import pandas as pd
from groq import Groq
import json


def extract_jokes_from_db():
    joke_ds = load_dataset("SeppeV/rated_jokes_dataset_from_jester", split="train")
    only_jokes = pd.DataFrame(joke_ds.select_columns("jokeText")).drop_duplicates()
    ds = Dataset.from_pandas(only_jokes, preserve_index=False)
    ds.push_to_hub("jester_jokes_extracted")


def add_thinking_steps_to_jokes():
    joke_ds = load_dataset("SeppeV/jester_jokes_extracted", split="train")

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    def map_dataset_to_add_thinking_steps(row):
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are going to give a detailed description of how jokes are constructed.
                                When given a joke to analyse, first provide the 2 main themes of the joke.
                                Then for each theme, generate a few associations that are relevant to that theme and the joke.
                                Then create a combination between 2 associations that are relevant for the joke and explain how this combination is funny.
                                Then extract the punchline from the joke.
                                After this extract the setup, which will be the rest of the joke.""",
                },
                {
                    "role": "user",
                    "content": f"Explain why the following joke is funny: {row['jokeText']}",
                },
            ],
            model="llama3-groq-70b-8192-tool-use-preview",
        )

        row["joke_reasoning_steps_llama70b"] = chat_completion.choices[
            0
        ].message.content

        return row

    joke_ds = joke_ds.map(map_dataset_to_add_thinking_steps)
    joke_ds.push_to_hub("SeppeV/jester_jokes_extracted")


def join_extracted_joke_database_with_extra_info_with_jester_db():
    jokes_with_extra_info = pd.DataFrame(
        load_dataset("SeppeV/jester_jokes_extracted", split="train")
    )
    jester_db = pd.DataFrame(
        load_dataset("SeppeV/rated_jokes_dataset_from_jester", split="train")
    )

    joined_db = pd.merge(jester_db, jokes_with_extra_info, on="jokeText")
    joined_db = Dataset.from_pandas(joined_db)
    joined_db.push_to_hub("SeppeV/rated_jokes_dataset_from_jester")


def create_csv_file_of_jokes():
    json.dump(
        load_dataset("SeppeV/jester_jokes_extracted", split="train")
        .select_columns("jokeText")
        .to_dict(),
        open("jester_jokes.json", "w"),
    )


if __name__ == "__main__":
    create_csv_file_of_jokes()
