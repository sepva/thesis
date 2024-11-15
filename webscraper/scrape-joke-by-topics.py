from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
import progressbar
from datasets import Dataset
import pickle


def get_soup(url):
    req = Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
    page = urlopen(req)
    html = page.read().decode("utf-8")
    return BeautifulSoup(html, "html.parser")


def get_all_topics():
    soup = get_soup("https://jokes.scoutlife.org/joke-by-topics/")
    return [
        item.string.split(" (")[0]
        for item in soup.find_all("a", attrs={"rel": "nofollow"})
        if item.string != None
    ]


def get_last_page_number(item):
    return (
        item.name == "a"
        and item.has_attr("class")
        and "page-numbers" in item.get("class")
        and (not "next" in item.get("class"))
    )


def get_all_text_jokes(item):
    return (
        item.name == "a"
        and item.has_attr("rel")
        and "bookmark" in item.get("rel")
        and item.find("img") == None
    )


def get_all_jokes_about_topic(topic):
    topic = topic.replace(" ", "-")
    soup = get_soup(f"https://jokes.scoutlife.org/topics/{topic}/")
    pages = soup.find_all(get_last_page_number)
    jokes = [item.text for item in soup.find_all(get_all_text_jokes)]
    for page in pages:
        page_number = page.text
        soup = get_soup(
            f"https://jokes.scoutlife.org/topics/{topic}/page/{page_number}/"
        )
        jokes.extend([item.text for item in soup.find_all(get_all_text_jokes)])

    return jokes


def get_all_jokes_by_topic():
    topics = get_all_topics()
    result_dict = {"topic": [], "joke": []}
    with progressbar.ProgressBar(max_value=len(topics)) as bar:
        for i, topic in enumerate(topics):
            try:
                jokes = get_all_jokes_about_topic(topic)
                for joke in jokes:
                    result_dict["topic"].append(topic)
                    result_dict["joke"].append(joke)

            except Exception as e:
                print(f"Exception with topic {topic}: {e}")
            bar.update(i)
    return result_dict


def save_jokes_to_huggingface(db_name, jokes):
    with open(f"{db_name}.pkl", "wb") as f:
        pickle.dump(jokes, f)
    ds = Dataset.from_dict(jokes)
    ds.push_to_hub(db_name)


if __name__ == "__main__":
    jokes = get_all_jokes_by_topic()
    save_jokes_to_huggingface("jokes_by_topic_from_scoutlife", jokes)
