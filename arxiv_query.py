import requests

url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1'

try:
    response = requests.get(url)
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    print(response.text)
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")