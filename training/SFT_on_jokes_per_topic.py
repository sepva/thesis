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


def get_datasets(dataset_id, tokenizer, joke_topic):
    print("Loading datasets...")
    ds = load_dataset(dataset_id, split="train")
    ds = ds.filter(lambda example: example["topic"] == joke_topic)
    joke_prompt = "You are a masterfull comedian. Tell me a classic Tom Swiftie joke:"

    def process(row):
        row["messages"] = [
            {"role": "user", "content": joke_prompt},
            {"role": "assistant", "content": row["joke"]},
        ]
        return row

    with PartialState().local_main_process_first():
        ds = ds.map(process)

    ds = ds.remove_columns(["topic", "joke"])

    split_db = ds.train_test_split(test_size=0.1)
    train_dataset = split_db["train"]
    eval_dataset = split_db["test"]
    return train_dataset, eval_dataset


def sft(model_id, dataset_id, trained_model_name, joke_topic):
    model = get_model(model_id)
    tokenizer = get_tokenizer(model_id)

    train_dataset, eval_dataset = get_datasets(dataset_id, tokenizer, joke_topic)

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


if __name__ == "__main__":
    sft(
        "mistralai/Mistral-7B-Instruct-v0.3",
        "SeppeV/jokes_by_topic_from_scoutlife",
        "mistral_instruct_ft_sft_tom_swiftie",
        "Tom Swiftie jokes",
    )
