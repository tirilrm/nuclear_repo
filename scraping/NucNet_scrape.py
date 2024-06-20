# NEI print test
import requests
import json
from bs4 import BeautifulSoup
import time

start_time = time.time()

f = open('urls/NucNet_urls.txt', 'r')
urls_raw = f.read()
urls = urls_raw.split('\n')

# First, give login credentials
login_url = 'https://www.nucnet.org/users/sign_in'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def scrape_NN_article(url, session):
    print(url)

    response = session.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    try:
        title = soup.find('h1', class_='mb-2 md:mb-6').get_text().split('/')[-1].strip()
    except AttributeError:
        title = None
    
    try:
        date_author = soup.find('p', class_='text-black text-sm sm:text-base md:text-lg').get_text().strip().split('\n')
        author = date_author[0][3:].strip()
        date = date_author[-1].strip()
    except AttributeError:
        date = None
        author = None
    
    try:
        captions = []
        figures = soup.find_all('figure')
        for figure in figures:
            figcaption = figure.find('figcaption')
            if figcaption:
                captions.append(figcaption.text)

    except AttributeError:
        captions = []
        
    try:
       text_divs = soup.find('article', class_='article-style').find_all('div')
       text = [p.get_text() for div in text_divs for p in div.find_all('p') if p.get_text() != '']
    except AttributeError:
        text = []
    
    text = text + captions

    return {
        'url': url,
        'magasine': 'NucNet',
        'title': title,
        'author': author,
        'date': date,
        'text': text
    }

# Log in
session = requests.Session()

login_page = session.get(login_url, headers=headers)
soup = BeautifulSoup(login_page.content, 'html.parser')

form_data = {}
hidden_fields = soup.find_all('input', type='hidden')
for field in hidden_fields:
    if field.has_attr('name'):
        form_data[field['name']] = field['value']

form_data['user[email]'] = 'tirilrm@gmail.com'
form_data['user[password]'] = 'password'

response = session.post(login_url, data=form_data, headers=headers)

if response.ok and 'logout' in response.text.lower():
    print('Login successful')
else:
    raise Exception('Login failed')

# Scrape
articles = [scrape_NN_article(url, session) for url in urls if url]

end_time = time.time()

print(f'Finished scrape, took {end_time-start_time:.2f} seconds')

with open('NucNet_articles_NEW.json', 'w', encoding='utf-8') as file:
    json.dump(articles, file, ensure_ascii=False, indent=4)