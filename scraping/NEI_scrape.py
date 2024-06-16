# NEI print test
import requests
import json
from bs4 import BeautifulSoup
import time

start_time = time.time()

f = open('NEI_urls.txt', 'r')
urls_raw = f.read()
urls = urls_raw.split('\n')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def scrape_NEI_article(url):
    print(url)

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    try:
        title = soup.find('h1', class_='article-header__title').get_text()
    except AttributeError:
        title = None
    
    try:
        date = soup.find('div', class_='article-header__content').find('span', class_='date-published').text
    except AttributeError:
        date = None
    
    try:
        author = soup.find('div', class_='article-header__content').find('span', class_='article-author').find('a')['href']
    except AttributeError:
        author = None
        
    try:
        article_all = soup.find('section', class_= 'article-content').get_text(separator="\n", strip=True)
        text_content = article_all.split('\n')
        unwanteds = ["share this article", "sign up", "image courtesy of", "partner content", "copy link", "share on", 'give your business an edge']
        text = []
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
        'magasine': 'Nuclear Engineering International',
        'title': title,
        'author': author,
        'date': date,
        'text': text
    }

articles = [scrape_NEI_article(url) for url in urls[2:] if url]

end_time = time.time()

print(f'Finished scrape, took {end_time-start_time:.2f} seconds')

with open('NEI_articles.json', 'w', encoding='utf-8') as file:
    json.dump(articles, file, ensure_ascii=False, indent=4)