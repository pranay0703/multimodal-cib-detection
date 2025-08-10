import os
import tweepy

def get_twitter_api():
    """
    Sets up and returns a Tweepy API object for the Twitter API.
    """
    # Note: For academic access, you might use a different authentication method.
    # This example uses App-only authentication (Bearer Token).
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise ValueError("TWITTER_BEARER_TOKEN environment variable not set.")

    api = tweepy.Client(bearer_token)
    return api

if __name__ == '__main__':
    try:
        api = get_twitter_api()
        print("Successfully authenticated with the Twitter API.")
        
        # Example: Get recent tweets containing the keyword "news"
        query = 'news -is:retweet'
        tweets = api.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=10)
        
        if tweets.data:
            for tweet in tweets.data:
                print(f"Tweet ID: {tweet.id}, Created at: {tweet.created_at}")
                print(f"Text: {tweet.text}
")
        else:
            print("No tweets found for the query.")

    except Exception as e:
        print(f"An error occurred: {e}")
