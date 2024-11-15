from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    single_squared_errors = ((logits - labels).flatten()**2).tolist()
    
    # Compute accuracy 
    # Saying that there is an accurate prediction if |pred-actual| < 2 => so sse < 4 
    accuracy = sum([1 for e in single_squared_errors if e < 4]) / len(single_squared_errors)
    
    return {"mse": mse, "mae": mae, "r2": r2, "accuracy": accuracy}
    
def test_model_to_ds(tokenizer, model):
    print("Get dataset")
    ds = load_dataset("SeppeV/rated_jokes_dataset_from_jester", split='train[:100]')
    ds = ds.filter(lambda row: row["userId"]==1)
    
    def transform_dataset(rows):
        input_texts = [f"User {userId} {jokeText}" for userId, jokeText in zip(rows["userId"], rows["jokeText"])]
        encoding = tokenizer(input_texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt").to(device)
        rows["pred"] = model(**encoding).logits
        rows["label"] = [float(rating) for rating in rows["rating"]]
        return rows
    
    ds = ds.map(transform_dataset, remove_columns=["userId", "rating", "jokeText", "__index_level_0__"], batched=True, batch_size=5)
    return ds

def test_user_proxy(model_id, tokenizer_id):
    print("Get tokenizer and model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    
    ds = test_model_to_ds(tokenizer, model)
    print(compute_metrics_for_regression(list([np.array(l) for l in ds[:62].values()])))
    
    
    
if __name__ == "__main__":
    test_user_proxy("SeppeV/bert_340M_ft_user1_pref", "bert-large-cased-whole-word-masking")