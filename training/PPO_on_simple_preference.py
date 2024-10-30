import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from accelerate import PartialState
from trl import (
    PPOConfig,
    PPOTrainer,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_model(model_id, reward_model=False):
    print("Getting model...")
    if reward_model:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
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
    ds = load_dataset(dataset_id, split=f"train[:100]")

    def process(row):
        row["input_ids"] = tokenizer.encode(row["question"])
        return row

    if map_dataset:
        with PartialState().local_main_process_first():
            ds = ds.map(process)

    ds = ds.remove_columns(["question", "A0", "A1", "A2", "A3"])

    split_db = ds.train_test_split(test_size=0.1)
    train_dataset = split_db["train"]
    eval_dataset = split_db["test"]
    return train_dataset, eval_dataset


def train_model_with_ppo(
    model_id,
    reward_model_id,
    dataset_id,
    trained_model_name,
    percentage_data=1,
    objective_function=None,
):
    model = get_model(model_id)
    ref_model = get_model(model_id)

    reward_model = get_model(reward_model_id, reward_model=True)
    value_model = get_model(reward_model_id, reward_model=True)

    tokenizer = get_tokenizer(model_id)

    train_dataset, eval_dataset = get_datasets(
        dataset_id,
        tokenizer,
        map_dataset=True,
        percentage_data=percentage_data,
        objective_function=objective_function,
    )

    training_args = PPOConfig(
        exp_name=trained_model_name,
        log_with="wandb",
        model_name=model_id,
        task_name=trained_model_name,
        batch_size=2,
        mini_batch_size=2,
    )

    ################
    # Training
    ################
    trainer = PPOTrainer(
        config=training_args,
        model=model,
        ref_model=ref_model,
        dataset=train_dataset,
        tokenizer=tokenizer,
    )
    print("Start Training!")
    trainer.train()
    model.save_pretrained(trained_model_name, push_to_hub=True)


if __name__ == "__main__":
    train_model_with_ppo(
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "SeppeV/reward_model_trained_on_a_preference_1pc_data",
        "SeppeV/q_and_as_with_SmolLM",
        "SmolLM_trained_on_a_preference_with_ppo_on_100rows_data",
        0.001,
    )
