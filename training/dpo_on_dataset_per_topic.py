import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import PartialState
from peft import LoraConfig
from trl import (
    DPOConfig,
    DPOTrainer,
    RichProgressCallback,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_model(model_id):
    print("Getting model...")
    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        quantization_config=bnb_config,
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
    dataset_id, tokenizer, map_dataset, percentage_data, filter_function, topic_to_learn
):
    print("Loading datasets...")
    full_ds = load_dataset(dataset_id, split=f"train[:{int(percentage_data*100)}%]")
    jokes_of_topic_to_learn = full_ds.filter(
        lambda example: example["topic"] == topic_to_learn
    )
    other_jokes = filter_function(full_ds).filter(
        lambda example: example["topic"] != topic_to_learn
    )
    prompt_for_joke = (
        "You are a masterfull comedian. Tell me a classic Tom Swiftie joke:"
    )
    prompt_in_chat = {"role": "user", "content": prompt_for_joke}

    def process(rows):
        rows["chosen"] = []
        rows["rejected"] = []
        rows["prompt"] = []
        for bad_joke in rows["joke"]:
            for joke in jokes_of_topic_to_learn["joke"]:
                rows["chosen"].append(
                    tokenizer.apply_chat_template(
                        [prompt_in_chat, {"role": "assistant", "content": joke}],
                        tokenize=False,
                    )
                )
                rows["rejected"].append(
                    tokenizer.apply_chat_template(
                        [prompt_in_chat, {"role": "assistant", "content": bad_joke}],
                        tokenize=False,
                    )
                )
                rows["prompt"].append(
                    tokenizer.apply_chat_template([prompt_in_chat], tokenize=False)
                )
        return rows

    if map_dataset:
        with PartialState().local_main_process_first():
            ds = other_jokes.map(
                process, batched=True, remove_columns=["topic", "joke"], batch_size=10
            )

    split_db = ds.train_test_split(test_size=0.1)
    train_dataset = split_db["train"]
    eval_dataset = split_db["test"]
    return train_dataset, eval_dataset


def dpo(
    model_id,
    dataset_id,
    trained_model_name,
    percentage_data,
    filter_function,
    topic_to_learn,
):
    model = get_model(model_id)
    # ref_model = get_model(model_id)

    tokenizer = get_tokenizer(model_id)

    train_dataset, eval_dataset = get_datasets(
        dataset_id,
        tokenizer,
        map_dataset=True,
        percentage_data=percentage_data,
        filter_function=filter_function,
        topic_to_learn=topic_to_learn,
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="MASKED_LM",
    )

    training_args = DPOConfig(
        trained_model_name,
        report_to="wandb",
        run_name=trained_model_name,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=True,
        bf16=False,
        optim="paged_adamw_32bit",
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
        ref_model=None,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[RichProgressCallback],
    )
    print("Start Training!")
    trainer.train()


def get_5_largest_joke_topics(dataset):
    topics = dataset["topic"]
    largest_topics = sorted(set(topics), key=topics.count)[-5:]
    largest_topics_jokes = dataset.filter(
        lambda example: example["topic"] in largest_topics
    )
    return largest_topics_jokes


if __name__ == "__main__":
    dpo(
        "mistralai/Mistral-7B-Instruct-v0.3",
        "SeppeV/jokes_by_topic_from_scoutlife",
        "mistralai_instruct_trained_on_jokes_by_topic_with_tom_swiftie_prompt_dpo",
        1,
        get_5_largest_joke_topics,
        "Tom Swiftie jokes",
    )
