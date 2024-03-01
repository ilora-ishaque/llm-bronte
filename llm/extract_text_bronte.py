from typing import Generator, Tuple
from pathlib import Path
import requests
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def extract(url: str, extraction_path: Path) -> None:
    """Extract text from an HTML URL and write it to a file.

    Args:
        url (str): URL of the HTML page containing the book text.
        extraction_path (Path): Path to save the extracted text.
    """
    LOGGER.info(f'Start extracting text from {url}')
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    chapters = extract_chapters(soup)
    text = extract_text_from_html(chapters)
    LOGGER.info(f'Finished extracting text from {url}')
    write_text_to_file(text, extraction_path)


def write_text_to_file(text: str, path: Path) -> None:
    """Write text to a file.

    Args:
        text (str): Text to be written to the file.
        path (Path): Path to save the file.
    """
    LOGGER.info(f'Start writing to {path}')
    with open(path, 'w') as f:
        f.write(text)
    LOGGER.info(f'Finished writing to {path}')


def extract_chapters(soup: BeautifulSoup) -> BeautifulSoup:
    """Extract chapters from the HTML content.

    Args:
        soup (BeautifulSoup): Parsed HTML content.

    Returns:
        BeautifulSoup: Parsed HTML content containing only the chapters.
    """
    # Extract all <div> elements with class 'chapter'
    chapters = soup.find_all('div', class_='chapter')
    return chapters


def extract_text_from_html(chapters: BeautifulSoup) -> str:
    """Extract text from HTML content.

    Args:
        chapters (BeautifulSoup): Parsed HTML content containing chapters.

    Returns:
        str: Extracted text.
    """
    text = ''
    for chapter in chapters:
        # Extract text from each chapter
        text += chapter.get_text() + '\n'
    return text


if __name__ == "__main__":
    # Example usage
    html_url = "https://www.gutenberg.org/cache/epub/9182/pg9182-images.html"
    output_path = Path("book_text.txt")
    extract(html_url, output_path)
