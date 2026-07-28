import requests

def fetch_blob_content(blob_url: str) -> str:
    """
    Fetches the content of a GitHub blob (base64 encoded).
    Safely decodes it into UTF-8 text.
    
    Returns:
        Decoded text (str), or an empty string on failure.
    """

    try:
        # GitHub blob API returns JSON containing base64 content
        response = requests.get(blob_url)
        response.raise_for_status()
        # data = response.json()
        data = response.text
        return data
    
    except Exception as e:
        print(f"Error fetching blob content from {blob_url}: {e}")
        return ""
