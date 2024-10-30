import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from trl import (
    DPOConfig,
    DPOTrainer,
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
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
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
        min_key = min(row, key=lambda k: objective_function(row, k))
        max_key = max(row, key=lambda k: objective_function(row, k))

        row["chosen"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": row[max_key]}], tokenize=False
        )
        row["rejected"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": row[min_key]}], tokenize=False
        )
        row["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}], tokenize=False
        )
        return row

    if map_dataset:
        with PartialState().local_main_process_first():
            ds = ds.map(process)

    ds = ds.remove_columns(["question"])

    split_db = ds.train_test_split(test_size=0.1)
    train_dataset = split_db["train"]
    eval_dataset = split_db["test"]
    return train_dataset, eval_dataset


def dpo_starter(
    model_id, dataset_id, trained_model_name, percentage_data=1, objective_function=None
):
    model = get_model(model_id)
    ref_model = get_model(model_id)

    tokenizer = get_tokenizer(model_id)

    train_dataset, eval_dataset = get_datasets(
        dataset_id,
        tokenizer,
        map_dataset=True,
        percentage_data=percentage_data,
        objective_function=objective_function,
    )

    training_args = DPOConfig(
        trained_model_name,
        report_to="wandb",
        run_name=trained_model_name,
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
    trainer = DPOTrainer(
        model,
        ref_model,
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
    dpo_starter(
        "SeppeV/SmolLM_trained_on_a_preference_with_sft_on_10pc_data",
        "SeppeV/q_and_as_with_SmolLM",
        "SmolLM_pretrained_with_sft_trained_with_1pc_data_on_a_preference_dpo",
        0.01,
        objective_function=get_a_freq,
    )
