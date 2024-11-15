from datasets import load_dataset
from scipy.stats import kstest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_ktest(shuffled_dataset, preference):
    fine_tuned_model_results = shuffled_dataset[f"{preference}_fine_tuned_model"]
    base_model_results = shuffled_dataset[f"{preference}_base_model"]

    return kstest(fine_tuned_model_results, base_model_results, alternative="less")


def plot_histogram(shuffled_dataset, title, preference):
    pd_shuffled_dataset = pd.DataFrame(shuffled_dataset)
    sns.histplot(
        pd_shuffled_dataset.melt(id_vars=["question"], value_vars=[f"{preference}_base_model", f"{preference}_fine_tuned_model"]),
        x="value",
        hue = "variable",
        palette={f"{preference}_fine_tuned_model": 'orange', f"{preference}_base_model": 'navy'},
        multiple="dodge",
        shrink=0.75,
        bins=20,
    )

    plt.title(title)
    plt.show()


if __name__ == "__main__":
    dataset_to_analyse = load_dataset(
        "SeppeV/test_complicated_preference_model_trained_on_10pc_data", split="test"
    )
    shuffled_dataset = dataset_to_analyse.shuffle()

    print(calculate_ktest(shuffled_dataset, preference="complicated_preference"))
    plot_histogram(
        shuffled_dataset,
        r"Histogram of mean hex value of letters in outputs of LLM's (with 10% of the total data)",
        preference="complicated_preference"
    )
