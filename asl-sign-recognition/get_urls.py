import requests
import json

urls = {}
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    # Try the most common ASL letter naming convention on Wikimedia
    filename = f"File:Sign_language_{letter}.svg"
    res = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&titles={filename}&prop=imageinfo&iiprop=url&format=json").json()
    pages = res.get('query', {}).get('pages', {})
    page = list(pages.values())[0]
    if 'imageinfo' in page:
        urls[letter] = page['imageinfo'][0]['url']
    else:
        urls[letter] = "NOT_FOUND"

print(json.dumps(urls, indent=2))
