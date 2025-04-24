import os
import sys
import requests
import time
import random
import re
import threading
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import arxiv
from pdfminer.high_level import extract_text

# Suppress pdfminer warnings (e.g. "CropBox missing from /Page, defaulting to MediaBox")
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Configuration
TARGET_SIZE = 25 * 1024 * 1024  # 25MB
REQUEST_DELAY = 1
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15'
]

# Initialize files if they don't exist
for f in ['L2.txt', 'L3.txt', 'L4.txt']:
    open(f, 'a').close()

def print_file_size(filename, label):
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"{label}: current size is {size_mb:.2f}MB")

# ================= L2: Gutenberg Books =================
def gutenberg_worker():
    book_id = 1
    session = requests.Session()
    while os.path.getsize('L2.txt') < TARGET_SIZE:
        try:
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}-images.html"
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for element in soup.select('.pg-header, .pg-boilerplate, nav, footer'):
                    element.decompose()
                text = soup.get_text(separator='\n', strip=True)
                with open('L2.txt', 'a', encoding='utf-8') as f:
                    f.write(f"=== Book {book_id} ===\n{text}\n\n")
                print(f"L2: Added book {book_id}")
                print_file_size('L2.txt', "L2")
            book_id += 1
            time.sleep(random.uniform(1, 2))
        except Exception as e:
            print(f"L2 Error: {str(e)}")
            time.sleep(5)

# ================= L3: News Articles =================
NEWS_SOURCES = [
    'https://www.reuters.com/news/archive',
    'https://apnews.com/hub/apf-topnews',
    'https://www.bbc.com/news'
]

def scrape_news_article(url, session):
    try:
        response = session.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        article = soup.find('article') or soup.find('div', class_=re.compile('article|content|main'))
        if article:
            for element in article.select('script, style, aside, nav'):
                element.decompose()
            return '\n'.join(p.get_text(strip=True) for p in article.find_all('p'))
        return None
    except Exception as e:
        print(f"Article error: {str(e)}")
        return None

def news_worker(source):
    session = requests.Session()
    session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
    while os.path.getsize('L3.txt') < TARGET_SIZE:
        try:
            response = session.get(source, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            links = [urljoin(source, a['href']) for a in soup.find_all('a', href=re.compile(r'/article/|/news/|-\d{4}-'))]
            for url in links[:10]:
                if os.path.getsize('L3.txt') >= TARGET_SIZE:
                    return
                content = scrape_news_article(url, session)
                if content and len(content) > 500:
                    with open('L3.txt', 'a', encoding='utf-8') as f:
                        f.write(f"URL: {url}\n{content}\n{'='*80}\n\n")
                    print(f"L3: Added article from {url}")
                    print_file_size('L3.txt', "L3")
                time.sleep(random.uniform(1, 2))
            time.sleep(random.uniform(5, 10))
        except Exception as e:
            print(f"News worker error: {str(e)}")
            time.sleep(15)

# ================= L4: Research Papers =================
ARXIV_HEADERS = {
    'Accept': 'application/pdf',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://arxiv.org/',
    'Connection': 'keep-alive'
}

def safe_pdf_download(paper):
    try:
        pdf_url = paper.pdf_url.replace('http:', 'https:')
        session = requests.Session()
        session.headers.update({
            **ARXIV_HEADERS,
            'User-Agent': random.choice(USER_AGENTS)
        })
        
        # Access abstract page to mimic a browser session
        session.get(paper.entry_id, timeout=10)
        time.sleep(random.uniform(1, 2))
        
        # Download PDF
        response = session.get(pdf_url, timeout=20)
        if response.status_code == 200:
            with open('temp.pdf', 'wb') as f:
                f.write(response.content)
            
            # Extract text from the PDF using pdfminer
            text = extract_text('temp.pdf')
            os.remove('temp.pdf')
            # Clean extracted text: remove hyphenated line breaks and join broken lines
            text = re.sub(r'-\n(\w)', r'\1', text)
            text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
            return text.strip()
        
        print(f"L4: Download failed ({response.status_code})")
        return None
    except Exception as e:
        print(f"PDF Error: {str(e)}")
        return None

def arxiv_worker():
    client = arxiv.Client(num_retries=3, delay_seconds=10)
    search = arxiv.Search(
        query="cat:cs.CL",
        max_results=300,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    try:
        for paper in client.results(search):
            if os.path.getsize('L4.txt') >= TARGET_SIZE:
                break
            
            print(f"\nL4: Processing {paper.title[:50]}...")
            
            full_text = None
            for attempt in range(3):
                full_text = safe_pdf_download(paper)
                if full_text:
                    break
                time.sleep(2 ** attempt)
            
            if not full_text:
                print("L4: Falling back to abstract")
                full_text = paper.summary
            
            with open('L4.txt', 'a', encoding='utf-8') as f:
                f.write(f"=== {paper.title} ===\n")
                f.write(f"Authors: {', '.join(a.name for a in paper.authors)}\n")
                f.write(f"Published: {paper.published.date()}\n")
                f.write(f"URL: {paper.entry_id}\n\n")
                f.write(full_text)
                f.write("\n\n" + "="*80 + "\n\n")
            
            print(f"L4: Added {len(full_text)//1024}KB")
            print_file_size('L4.txt', "L4")
            time.sleep(random.uniform(15, 30))
            
    except Exception as e:
        print(f"arXiv Error: {str(e)}")

if __name__ == "__main__":
    # Start workers as daemon threads.
    threading.Thread(target=gutenberg_worker, daemon=True).start()
    for source in NEWS_SOURCES:
        threading.Thread(target=news_worker, args=(source,), daemon=True).start()
    threading.Thread(target=arxiv_worker, daemon=True).start()
    
    # Main loop checks if all files have reached the target size.
    while True:
        l2_size = os.path.getsize('L2.txt')
        l3_size = os.path.getsize('L3.txt')
        l4_size = os.path.getsize('L4.txt')
        
        if l2_size >= TARGET_SIZE and l3_size >= TARGET_SIZE and l4_size >= TARGET_SIZE:
            print("All files have reached the target size. Exiting program.")
            sys.exit(0)
        time.sleep(5)
