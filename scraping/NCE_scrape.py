# NCE print test
import requests
import json
from bs4 import BeautifulSoup
import time

start_time = time.time()

f = open('NCE_urls.txt', 'r')
urls_raw = f.read()
urls = urls_raw.split('\n')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def scrape_NCE_article(url):
    print(url)

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    try:
        title = soup.find('h1', class_='name entry-title').text
    except AttributeError:
        title = None
    
    try:
        date = soup.find('span', class_='tie-date').get_text()
    except AttributeError:
        date = None
    
    try:
        author = soup.find('a', class_='author url fn').text
    except AttributeError:
        author = None

    try:
        article_all = soup.find('div', class_= 'entry').get_text(separator="\n", strip=True)
        text_content = article_all.split('\n')
        text = []
        unwanteds = ['to receive new civil engineer\'s', 'like what you\'ve read?']
        if text_content:
            for line in text_content:
                if not any(unwanted in line.lower() \
                        for unwanted in unwanteds) \
                            and len(line) > 25:
                    text.append(line)
    except AttributeError:
        text = None
    return {
        'url': url,
        'magasine': 'New Civil Engineer',
        'title': title,
        'author': author,
        'date': date,
        'text': text
    }

articles = [scrape_NCE_article(url) for url in urls if url]

end_time = time.time()

print(f'Finished scrape, took {end_time-start_time} seconds')

with open('NCE_articles.json', 'w', encoding='utf-8') as file:
    json.dump(articles, file, ensure_ascii=False, indent=4)