from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import itertools
import matplotlib.patches as mpatches
import torch


def get_n_largest_topics_dataset(n):
    dataset = load_dataset("SeppeV/jokes_by_topic_from_scoutlife", split="train")
    topics = dataset["topic"]
    largest_topics = sorted(set(topics), key=topics.count)[-n:]
    largest_topics_jokes = dataset.filter(
        lambda example: example["topic"] in largest_topics
    )

    return largest_topics_jokes


def get_jokes_from_topic(topic):
    dataset = load_dataset("SeppeV/jokes_by_topic_from_scoutlife", split="train")
    jokes_from_topic = dataset.filter(lambda example: example["topic"] == topic)
    return jokes_from_topic


def plot_embeddings(dataset, fine_tuned_dataset_id, base_model_dataset_id, title):
    fine_tuned_dataset = load_dataset(fine_tuned_dataset_id, split="train")
    base_model_dataset = load_dataset(base_model_dataset_id, split="train")
    topics = dataset["topic"]
    color_cycle = ["r", "b", "g", "y", "m", "c", "k"]
    possible_colors = itertools.cycle(color_cycle)
    last_topic = ""
    colors = []
    for topic in topics:
        if topic != last_topic:
            last_color = next(possible_colors)

        colors.append(last_color)
        last_topic = topic

    dataset = dataset.remove_columns(["topic"])
    combined_ds = concatenate_datasets(
        [dataset, fine_tuned_dataset, base_model_dataset]
    )
    colors.extend([next(possible_colors)] * len(fine_tuned_dataset["joke"]))
    colors.extend([next(possible_colors)] * len(base_model_dataset["joke"]))

    result = calculate_embeddings_for_plot(combined_ds)

    topics_and_jokes_from_models = sorted(set(topics))
    topics_and_jokes_from_models.extend(["fine_tuned_model_jokes", "base_model_jokes"])

    add_legend(color_cycle, topics_and_jokes_from_models)
    plt.title(title)
    plt.scatter(result[:, 0], result[:, 1], c=colors, marker="o")
    plt.show()


def add_legend(colors, topics):
    handles = []
    for color, topic in zip(colors, topics):
        patch = mpatches.Patch(color=color, label=topic)
        handles.append(patch)
    plt.legend(handles=handles)


def calculate_embeddings_for_plot(dataset):
    model = SentenceTransformer("deepset/gbert-base")
    embeddings = model.encode(dataset["joke"])
    result = TSNE().fit_transform(embeddings)
    return result


def calculate_similarity(dataset_id, topic):
    output_jokes = load_dataset(dataset_id, split="train")
    tom_swiftie_jokes = get_jokes_from_topic(topic)

    model = SentenceTransformer("deepset/gbert-base")
    embeddings1 = model.encode(output_jokes["joke"])
    embeddings2 = model.encode(tom_swiftie_jokes["joke"])
    similarities = model.similarity(embeddings1, embeddings2)
    return similarities


def plot_similarity(dataset_id, topic, alignment, c):
    sims = calculate_similarity(dataset_id, topic)
    bins = 100
    hist = torch.histc(sims, bins=bins, min=0, max=1)
    x = [n / bins for n in range(bins)]
    plt.bar(x, hist.tolist(), width=alignment / (2 * bins), align="edge", color=c)


if __name__ == "__main__":
    # jokes = get_n_largest_topics_dataset(5)
    # plot_embeddings(
    #     jokes,
    #     "SeppeV/mistral_instruct_sft_dpo_joke_outputs_with_tom_swiftie_prompt_and_context",
    #     "SeppeV/mistral_instruct_joke_outputs_with_tom_swiftie_prompt_and_context",
    #     "Embeddings (deepset/gbert-base) of different joke topics with trained topic in prompt (Mistral+dpo+sft with context)",
    # )
    plot_similarity(
        "SeppeV/mistral_instruct_ft_dpo_joke_outputs_with_tom_swiftie_prompt_and_context",
        "Tom Swiftie jokes",
        -1,
        "b",
    )
    plot_similarity(
        "SeppeV/mistral_instruct_sft_dpo_joke_outputs_with_tom_swiftie_prompt_and_context",
        "Tom Swiftie jokes",
        1,
        "r",
    )
    add_legend(
        ["b", "r"], ["fine-tuned model with dpo", "fine-tuned model with sft+dpo"]
    )
    plt.title("Similarity of Tom Swiftie jokes between fine-tuned models")
    plt.show()
