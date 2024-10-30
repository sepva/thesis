from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn as nn
import torch
from transformers import AutoModel


class UserPreferenceModel(nn.Module):
    def __init__(self, base_model_id, tokenizer):
        super(UserPreferenceModel, self).__init__()

        self.base_model = AutoModel.from_pretrained(base_model_id)
        self.tokenizer = tokenizer
        self.dropout1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.8)
        self.linear2 = nn.Linear(512, 1)

    def resize_token_embeddings(self, size):
        self.base_model.resize_token_embeddings(size)

    def forward(self, input_ids, *args, **kwargs):
        input_ids = torch.IntTensor(input_ids)
        outputs = self.base_model(input_ids)
        outputs = self.dropout1(outputs[1])
        outputs = self.linear1(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.linear2(outputs)

        return outputs


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten() ** 2).tolist()

    # Compute accuracy
    # Saying that there is an accurate prediction if |pred-actual| < 2 => so sse < 4
    accuracy = sum([1 for e in single_squared_errors if e < 4]) / len(
        single_squared_errors
    )

    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}


def get_dataset(tokenizer, model):
    print("Get dataset")
    ds = load_dataset("SeppeV/rated_jokes_dataset_from_jester", split="train[:10%]")

    def transform_dataset(row):
        label = row["rating"]
        user_token = f"User {row['userId']}"
        input_text = f"{user_token} {row['jokeText']}"
        tokenizer.add_tokens([user_token])
        model.resize_token_embeddings(len(tokenizer))
        row["input_ids"] = tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        ).input_ids
        row["label"] = float(label)
        return row

    ds = ds.map(
        transform_dataset,
        remove_columns=["userId", "rating", "jokeText", "__index_level_0__"],
    )

    split_db = ds.train_test_split(test_size=0.1)
    train_dataset = split_db["train"]
    eval_dataset = split_db["test"]
    return train_dataset, eval_dataset


def train_user_preference_proxy(base_model, trained_model_name):
    print("Get tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = UserPreferenceModel(base_model, tokenizer)

    train_ds, eval_ds = get_dataset(tokenizer, model)

    training_args = TrainingArguments(
        output_dir=trained_model_name,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        report_to="wandb",
        run_name=trained_model_name,
        per_device_eval_batch_size=8,
        num_train_epochs=16,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        weight_decay=0.01,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics_for_regression,
    )
    trainer.train()


if __name__ == "__main__":
    train_user_preference_proxy(
        "bert-large-cased-whole-word-masking",
        "bert_340M_ft_first_10pc_pref_2_lin_layers_usertokens",
    )
