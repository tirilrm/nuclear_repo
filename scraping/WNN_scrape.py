# WNN print test
import requests
import json
from bs4 import BeautifulSoup

f = open('WNN_urls.txt', 'r')
urls_raw = f.read()
urls = urls_raw.split('\n')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def scrape_WNN_article(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    article_all = soup.find('div', class_= 'ArticleBody').get_text(separator="\n", strip=True)
    text_content = article_all.split('\n')
    title = text_content[0]
    date = soup.find('div', class_='col-md-8 ArticleBody').find('p').text

    unwanteds = ['related topics']
    text=[]
    if text_content:
        for line in text_content:
            if not any(unwanted in line.lower() \
                    for unwanted in unwanteds) \
                        and len(line) > 25 and line not in title:
                text.append(line)
    return {
        'url': url,
        'title': title,
        'author': None,
        'date': date,
        'text': text
    }

articles = [scrape_WNN_article(url) for url in urls]

with open('WNN_articles.json', 'w', encoding='utf-8') as file:
    json.dump(articles, file, ensure_ascii=False, indent=4)