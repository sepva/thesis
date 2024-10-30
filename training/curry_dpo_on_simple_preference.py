import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from trl import DPOConfig, DPOTrainer, RichProgressCallback

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


def get_datasets(dataset_id, tokenizer, index, map_dataset=True):
    print("Loading datasets...")
    ds = load_dataset(dataset_id, split="train[:10%]")

    def get_a_freq(row, column):
        text = row.get(column)
        length = len(text)
        return text.count("a") / length if length > 0 else 0

    def process(row, indx=None):
        sorted_rows = sorted(row, key=lambda k: get_a_freq(row, k))
        sorted_rows.remove("question")
        rejected_key = sorted_rows[index if indx == None else indx]
        chosen_key = sorted_rows[-1]

        row["chosen"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": row[chosen_key]}], tokenize=False
        )
        row["rejected"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": row[rejected_key]}], tokenize=False
        )
        row["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}], tokenize=False
        )
        return row

    def process_eval_dataset(row):
        return process(row, indx=random.randint(0, 2))

    split_db = ds.train_test_split(test_size=0.1)
    train_dataset = split_db["train"]
    eval_dataset = split_db["test"]

    if map_dataset:
        with PartialState().local_main_process_first():
            train_dataset = train_dataset.map(process)
            eval_dataset = eval_dataset.map(process_eval_dataset)

    train_dataset.remove_columns(["question"])
    eval_dataset.remove_columns(["question"])

    return train_dataset, eval_dataset


def curry_dpo(model_id, dataset_id, new_name):
    for i in range(3):
        if i > 0:
            model_id = f"{new_name}_iter_{i-1}"

        model = get_model(model_id)
        ref_model = get_model(model_id)

        tokenizer = get_tokenizer(model_id)

        train_dataset, eval_dataset = get_datasets(
            dataset_id, tokenizer, i, map_dataset=True
        )

        new_iter_model_name = f"{new_name}_iter_{i}"

        training_args = DPOConfig(
            new_iter_model_name,
            report_to="wandb",
            run_name=new_iter_model_name,
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


if __name__ == "__main__":
    curry_dpo(
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "SeppeV/q_and_as_with_SmolLM",
        "SeppeV/SmolLM_trained_with_curry_dpo_10pc_data_fixed_eval_set",
    )
