import requests

def fetch_agmarknet_news(place, crop):
    api_url = "https://api.agmarknet.gov/news"  # Replace with the actual API URL

    try:
        response = requests.get(api_url, timeout=10)  # Set a timeout to prevent hanging requests

        if response.status_code == 200:
            articles = response.json().get("articles", [])

            if not articles:
                print("No articles found in API response.")
                return []

            # Convert crop name to lowercase for case-insensitive filtering
            crop_lower = crop.lower()

            # Filter news articles related to the crop
            filtered_articles = [
                article for article in articles
                if crop_lower in article.get("title", "").lower() or crop_lower in article.get("description", "").lower()
            ]

            return filtered_articles[:5]  # Return only top 5 news articles

        else:
            print(f"Error: Received status code {response.status_code} from Agmarknet API")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Agmarknet news: {e}")
        return []
