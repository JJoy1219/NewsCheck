"""
Improved News Scraper - Fixes URL issues and adds alternative sources
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Optional
import re
from urllib.parse import urljoin, urlparse

class ImprovedNewsScraper:
    def __init__(self, delay=2.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(Advertisement|ADVERTISEMENT|Subscribe|Read more)', '', text)
        return text.strip()

    def fix_url(self, url: str, base_url: str) -> str:
        """Fix malformed URLs"""
        if url.startswith('//'):
            url = 'https:' + url
        elif not url.startswith('http'):
            url = urljoin(base_url, url)
        return url

    # AP News - working well
    def scrape_ap_articles(self, num_articles=50) -> List[Dict]:
        articles = []
        hub_url = "https://apnews.com/hub/us-news"
        soup = self.get_soup(hub_url)
        if not soup:
            return articles

        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/article/' in href:
                full_url = urljoin(hub_url, href)
                links.add(full_url)

        print(f"Found {len(links)} AP article links, scraping {min(num_articles, len(links))}...")

        for i, url in enumerate(list(links)[:num_articles]):
            article = self.scrape_generic_article(url, 'AP News', 'center')
            if article:
                articles.append(article)
                print(f"  [{i+1}/{num_articles}] {article['title'][:60]}...")
            time.sleep(self.delay)

        return articles

    # Fox News - FIXED URL handling
    def scrape_fox_articles(self, num_articles=50) -> List[Dict]:
        articles = []
        hub_url = "https://www.foxnews.com/politics"
        soup = self.get_soup(hub_url)
        if not soup:
            return articles

        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Fix malformed URLs
            if href.startswith('//'):
                href = 'https:' + href
            elif not href.startswith('http'):
                href = urljoin(hub_url, href)

            # Only keep actual article links
            if 'foxnews.com' in href and '/politics/' in href and '/category/' not in href:
                links.add(href)

        print(f"Found {len(links)} Fox article links, scraping {min(num_articles, len(links))}...")

        for i, url in enumerate(list(links)[:num_articles]):
            article = self.scrape_generic_article(url, 'Fox News', 'right')
            if article:
                articles.append(article)
                print(f"  [{i+1}/{num_articles}] {article['title'][:60]}...")
            time.sleep(self.delay)

        return articles

    # NY Post - Alternative right-wing source
    def scrape_nypost_articles(self, num_articles=50) -> List[Dict]:
        articles = []
        hub_url = "https://nypost.com/news/"
        soup = self.get_soup(hub_url)
        if not soup:
            return articles

        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'nypost.com' in href and '/2024/' in href or '/2025/' in href:
                links.add(href)

        print(f"Found {len(links)} NY Post article links, scraping {min(num_articles, len(links))}...")

        for i, url in enumerate(list(links)[:num_articles]):
            article = self.scrape_generic_article(url, 'NY Post', 'right')
            if article:
                articles.append(article)
                print(f"  [{i+1}/{num_articles}] {article['title'][:60]}...")
            time.sleep(self.delay)

        return articles

    # CNN - Improved link extraction
    def scrape_cnn_articles(self, num_articles=50) -> List[Dict]:
        articles = []
        hub_url = "https://www.cnn.com/politics"
        soup = self.get_soup(hub_url)
        if not soup:
            return articles

        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(hub_url, href)
            # CNN article URLs have year in them
            if 'cnn.com' in full_url and ('/2024/' in full_url or '/2025/' in full_url):
                # Skip video/gallery pages
                if '/video/' not in full_url and '/gallery/' not in full_url:
                    links.add(full_url)

        print(f"Found {len(links)} CNN article links, scraping {min(num_articles, len(links))}...")

        for i, url in enumerate(list(links)[:num_articles]):
            article = self.scrape_generic_article(url, 'CNN', 'left')
            if article:
                articles.append(article)
                print(f"  [{i+1}/{num_articles}] {article['title'][:60]}...")
            time.sleep(self.delay)

        return articles

    # HuffPost - Alternative left-wing source
    def scrape_huffpost_articles(self, num_articles=50) -> List[Dict]:
        articles = []
        hub_url = "https://www.huffpost.com/news/politics"
        soup = self.get_soup(hub_url)
        if not soup:
            return articles

        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(hub_url, href)
            if 'huffpost.com' in full_url and '/entry/' in full_url:
                links.add(full_url)

        print(f"Found {len(links)} HuffPost article links, scraping {min(num_articles, len(links))}...")

        for i, url in enumerate(list(links)[:num_articles]):
            article = self.scrape_generic_article(url, 'HuffPost', 'left')
            if article:
                articles.append(article)
                print(f"  [{i+1}/{num_articles}] {article['title'][:60]}...")
            time.sleep(self.delay)

        return articles

    # The Hill - Alternative center source (since Reuters is blocked)
    def scrape_hill_articles(self, num_articles=50) -> List[Dict]:
        articles = []
        hub_url = "https://thehill.com/politics/"
        soup = self.get_soup(hub_url)
        if not soup:
            return articles

        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(hub_url, href)
            if 'thehill.com' in full_url and '/policy/' in full_url or '/politics/' in full_url:
                links.add(full_url)

        print(f"Found {len(links)} The Hill article links, scraping {min(num_articles, len(links))}...")

        for i, url in enumerate(list(links)[:num_articles]):
            article = self.scrape_generic_article(url, 'The Hill', 'center')
            if article:
                articles.append(article)
                print(f"  [{i+1}/{num_articles}] {article['title'][:60]}...")
            time.sleep(self.delay)

        return articles

    def scrape_generic_article(self, url: str, source: str, label: str) -> Optional[Dict]:
        """Generic article scraper that works for most news sites"""
        soup = self.get_soup(url)
        if not soup:
            return None

        try:
            # Try multiple title selectors
            title = ""
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                title = title_elem.get_text()

            # Extract all paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            text = self.clean_text(text)

            # Minimum length check
            if len(text) < 200:
                return None

            return {
                'source': source,
                'url': url,
                'title': title.strip(),
                'text': text,
                'label': label,
                'scraped_at': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error parsing {url}: {e}")
            return None


def main():
    scraper = ImprovedNewsScraper(delay=1.5)
    all_articles = []

    # Target: ~60 articles per label for balanced dataset

    print("\n=== CENTER SOURCES ===")
    print("\nScraping AP News...")
    all_articles.extend(scraper.scrape_ap_articles(num_articles=40))

    print("\nScraping The Hill...")
    all_articles.extend(scraper.scrape_hill_articles(num_articles=40))

    print("\n=== RIGHT SOURCES ===")
    print("\nScraping Fox News...")
    all_articles.extend(scraper.scrape_fox_articles(num_articles=40))

    print("\nScraping NY Post...")
    all_articles.extend(scraper.scrape_nypost_articles(num_articles=40))

    print("\n=== LEFT SOURCES ===")
    print("\nScraping CNN...")
    all_articles.extend(scraper.scrape_cnn_articles(num_articles=40))

    print("\nScraping HuffPost...")
    all_articles.extend(scraper.scrape_huffpost_articles(num_articles=40))

    # Save
    df = pd.DataFrame(all_articles)
    df.to_csv('news_articles.csv', index=False)

    print(f"\n{'='*60}")
    print("SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"Total articles: {len(all_articles)}")
    print(f"\nBy label:\n{df['label'].value_counts()}")
    print(f"\nBy source:\n{df['source'].value_counts()}")

    # Check balance
    label_counts = df['label'].value_counts()
    min_count = label_counts.min()
    max_count = label_counts.max()

    if max_count / min_count > 2:
        print(f"\n⚠️  WARNING: Dataset imbalanced!")
        print(f"   Ratio: {max_count/min_count:.1f}x difference")
        print(f"   Consider collecting more articles from underrepresented sources")
    else:
        print(f"\n✓ Dataset reasonably balanced")

    print(f"\nSaved to: news_articles.csv")


if __name__ == "__main__":
    main()