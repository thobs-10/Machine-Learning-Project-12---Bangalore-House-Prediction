import requests
import pandas as pd
from dagster import asset, get_dagster_logger # import the `dagster` library

@asset # add the asset decorator to tell Dagster this is an asset
def topstory_ids(): # turn it into a function
    newstories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    top_new_story_ids = requests.get(newstories_url).json()[:100]
    return top_new_story_ids # return the data

@asset
def topstories(topstory_ids):  # this asset is dependent on topstory_ids

    logger = get_dagster_logger()

    results = []
    for item_id in topstory_ids:
        item = requests.get(
            f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
        ).json()
        results.append(item)

        if len(results) % 20 == 0:
            logger.info(f"Got {len(results)} items so far.")


    df = pd.DataFrame(results)

    return df

@asset
def most_frequent_words(topstories):
    stopwords = ["a", "the", "an", "of", "to", "in", "for", "and", "with", "on", "is"]

    # loop through the titles and count the frequency of each word
    word_counts = {}
    for raw_title in topstories["title"]:
        title = raw_title.lower()
        for word in title.split():
            cleaned_word = word.strip(".,-!?:;()[]'\"-")
            if cleaned_word not in stopwords and len(cleaned_word) > 0:
                word_counts[cleaned_word] = word_counts.get(cleaned_word, 0) + 1

    # Get the top 25 most frequent words
    top_words = {
        pair[0]: pair[1]
        for pair in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:25]
    }

    return top_words

@asset
def another_func(topstory_ids,most_frequent_words):
    logger = get_dagster_logger()

    logger.info(f"Got {len(topstory_ids)} ids so far.")
    logger.info(f"Got {len(most_frequent_words)} ids so far.")
