import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from trl import (
    SFTConfig,
    SFTTrainer,
    RichProgressCallback,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_model(model_id):
    print("Getting model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    return model


def get_tokenizer(model_id):
    print("Getting tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    return tokenizer


def get_datasets(
    dataset_id, tokenizer, map_dataset=True, percentage_data=1, objective_function=None
):
    print("Loading datasets...")
    ds = load_dataset(dataset_id, split=f"train[:{int(percentage_data*100)}%]")

    def process(row):
        max_key = max(row, key=lambda k: objective_function(row, k))

        row["messages"] = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row[max_key]},
        ]
        return row

    if map_dataset:
        with PartialState().local_main_process_first():
            ds = ds.map(process)

    ds = ds.remove_columns(["question", "A0", "A1", "A2", "A3"])

    split_db = ds.train_test_split(test_size=0.1)
    train_dataset = split_db["train"]
    eval_dataset = split_db["test"]
    return train_dataset, eval_dataset


def sft(
    model_id, dataset_id, trained_model_name, percentage_data=1, objective_function=None
):
    model = get_model(model_id)
    tokenizer = get_tokenizer(model_id)

    train_dataset, eval_dataset = get_datasets(
        dataset_id,
        tokenizer,
        map_dataset=True,
        percentage_data=percentage_data,
        objective_function=objective_function,
    )

    training_args = SFTConfig(
        trained_model_name,
        report_to="wandb",
        run_name=trained_model_name,
        packing=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_total_limit=2,
        eval_strategy="epoch",
        push_to_hub=True,
        hub_private_repo=True,
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[RichProgressCallback],
    )
    print("Start Training!")
    trainer.train()


def get_a_freq(row, column):
    text = row.get(column)
    length = len(text)
    return text.count("a") / length if length > 0 else 0


def get_more_complicated_preference(row, column):
    text = row.get(column)
    length = len(text)
    text_score = sum([ord(letter) for letter in text])
    return text_score / length if length > 0 else 0


def the_preference(row, column):
    text = row.get(column)
    length = len(text)
    text_score = text.count("The") + text.count("the")
    return text_score / length if length > 0 else 0


if __name__ == "__main__":
    sft(
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "SeppeV/q_and_as_with_SmolLM",
        "SmolLM_trained_on_a_preference_with_sft_on_10pc_data",
        0.1,
        objective_function=get_a_freq,
    )
