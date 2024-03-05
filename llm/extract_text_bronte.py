from typing import Generator, Tuple
from pathlib import Path
import requests
import logging
import json
from tqdm import tqdm
from bs4 import BeautifulSoup

from llm import config

# import sys    

# path_to_module = 'content/llm-bronte/llm/config.py'
# sys.path.append(path_to_module)
# import config


# logging.basicConfig(level=logging.INFO)
# LOGGER = logging.getLogger(__name__)


# def extract(url: str, extraction_path: Path) -> None:
#     """Extract text from an HTML url and store the content of <div> elements with class 'chapter' in a JSONL file.

#     Args:
#         url (str): URL of the HTML page
#         extraction_path (Path): Path to save the JSONL file
#     """
#     LOGGER.info(f'Start extracting content from {url}')
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     chapters = soup.find_all('div', class_='chapter')
#     pages = ((i, chapter.get_text()) for i, chapter in enumerate(chapters, start=1))
#     LOGGER.info(f'Finished extracting content from {url}')
#     to_jsonl(pages=pages, path=extraction_path)


# def to_jsonl(pages: Generator[Tuple[int, str], None, None], path: Path) -> None:
#     LOGGER.info(f'Start writing to {path}')
#     with open(path, 'w') as f:
#         for page_number, text in tqdm(pages):
#             dict_page = {page_number: text}
#             json.dump(dict_page, f)
#             f.write('\n')
#     LOGGER.info(f'Finished writing to {path}')


# if __name__ == "__main__":
#     extract(url=config.url, extraction_path=config.extraction_path)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def extract(url_list: list, extraction_path: Path) -> None:
    """Extract text from HTML urls and store the content of <div> elements with class 'chapter' in a JSONL file.

    Args:
        url_list (list): List of URLs of the HTML pages
        extraction_path (Path): Path to save the JSONL file
    """
    LOGGER.info('Start extracting content from URLs')
    all_pages = []
    for url in url_list:
        LOGGER.info(f'Start extracting content from {url}')
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        chapters = soup.find_all('div', class_='chapter')
        pages = ((i, chapter.get_text()) for i, chapter in enumerate(chapters, start=1))
        all_pages.extend(pages)
        LOGGER.info(f'Finished extracting content from {url}')

    LOGGER.info(f'Start writing to {extraction_path}')
    with open(extraction_path, 'w') as f:
        for page_number, text in tqdm(all_pages):
            dict_page = {page_number: text}
            json.dump(dict_page, f)
            f.write('\n')
    LOGGER.info(f'Finished writing to {extraction_path}')


if __name__ == "__main__":
    urls = [config.url1, config.url2, config.url3]  # Add URLs as required
    extract(url_list=urls, extraction_path=config.extraction_path)