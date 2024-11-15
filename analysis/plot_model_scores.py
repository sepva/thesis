from datasets import load_dataset
import matplotlib.pyplot as plt
import statistics


def get_scores(dataset):
    dataset = load_dataset(dataset, split="train")
    scores = [s[0] for s in dataset["score"]]
    return scores


def plot_scores(base_model_ds, ft_model_ds):
    base_model_scores = get_scores(base_model_ds)
    ft_model_scores = get_scores(ft_model_ds)
    print("Base model mean:", statistics.mean(base_model_scores))
    print("Base model median:", statistics.median(base_model_scores))
    print("Ft model mean:", statistics.mean(ft_model_scores))
    print("Ft model median:", statistics.median(ft_model_scores))
    plt.hist(base_model_scores, 20, label="Scores of base model", alpha=0.75)
    plt.hist(ft_model_scores, 20, label="Scores of fine-tuned model", alpha=0.75)
    plt.title("Scores of base and ft model trained for 1000 datapoints with DPO")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_scores(
        "SeppeV/results_joke_gen_mistral_base_pe_judge_bert340_1000_random_user",
        "SeppeV/results_joke_gen_mistral_ft_dpo_pe_1000_judge_bert340_1000_random_user",
    )
