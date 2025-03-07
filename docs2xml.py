#!/usr/bin/env python3
"""
docs2xml: Crawl website documentation into XML, optionally chunk+embed in ChromaDB.

Usage:
    python docs2xml.py crawl <start_url> [options]
    python docs2xml.py embed <xml_file> [options]
    python docs2xml.py version

Example:
    # Crawl and immediately chunk+embed to Chroma
    python docs2xml.py crawl https://docs.example.com --chunk-and-embed --embedding-function=openai \
        --openai-api-key=sk-... --chunk-size=512 --chroma-collection="example_docs"

    # Embed an existing docs_archive.xml file
    python docs2xml.py embed docs_archive.xml --embedding-function=huggingface \
        --huggingface-model-name="sentence-transformers/all-MiniLM-L6-v2" \
        --chunk-size=512 --chroma-collection="my_docs"
"""

import sys

# Insert default subcommand "crawl" if none is provided
if len(sys.argv) <= 1 or sys.argv[1] not in ['crawl', 'embed', 'version']:
    sys.argv.insert(1, 'crawl')

import argparse
import asyncio
import logging
import re
import time
import xml.dom.minidom
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

# Basic dependencies for crawling
try:
    import aiohttp
    import bs4
    from bs4 import BeautifulSoup
    from lxml import etree
    import readability
    from readability import Document
    import langdetect
    from langdetect import detect_langs
    import tqdm
    import colorama
    from colorama import Fore, Style
    from urllib.robotparser import RobotFileParser
except ImportError as e:
    print(f"Error: Required dependency not found: {e}")
    print("Please install all required dependencies with:")
    print("  pip install aiohttp beautifulsoup4 lxml readability-lxml langdetect tqdm colorama python-robots")
    sys.exit(1)

colorama.init()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("doc_crawler")


###############################################################################
#                            CRAWLER IMPLEMENTATION                           #
###############################################################################
class DocCrawler:
    """Main crawler class to handle website documentation crawling and XML conversion."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the crawler with command line arguments."""
        self.start_url = args.start_url
        self.output_file = args.output_file
        self.max_depth = args.max_depth
        self.max_pages = args.max_pages
        self.user_agent = args.user_agent
        self.delay = args.delay
        self.include_pattern = re.compile(args.include_pattern) if args.include_pattern != 'None' else None
        self.exclude_pattern = re.compile(args.exclude_pattern) if args.exclude_pattern != 'None' else None
        self.timeout = args.timeout
        self.verbose = args.verbose
        self.include_images = args.include_images
        self.include_code = args.include_code
        self.extract_headings = args.extract_headings
        self.follow_links = args.follow_links
        self.clean_html = args.clean_html
        self.strip_js = args.strip_js
        self.strip_css = args.strip_css
        self.strip_comments = args.strip_comments
        self.respect_robots = args.robots_txt
        self.concurrency = args.concurrency
        self.restrict_path = args.restrict_path

        # For chunk+embed
        self.chunk_and_embed = args.chunk_and_embed
        self.chunk_size = args.chunk_size
        self.chunk_overlap = args.chunk_overlap
        self.chunk_by_chars = args.chunk_by_chars
        self.ignore_content_boundaries = args.ignore_content_boundaries
        self.embedding_function_name = args.embedding_function
        self.openai_api_key = args.openai_api_key
        self.cohere_api_key = args.cohere_api_key
        self.huggingface_model_name = args.huggingface_model_name
        self.chroma_collection = args.chroma_collection
        self.chroma_persist_dir = args.chroma_persist_dir
        self.output_chunks = args.output_chunks

        # State
        self.visited_urls: Set[str] = set()
        self.pages_crawled = 0
        self.failed_urls: List[Tuple[str, str]] = []
        self.domain = urlparse(self.start_url).netloc
        
        parsed_start = urlparse(self.start_url)
        self.start_url_domain = parsed_start.netloc
        self.start_url_path = parsed_start.path or "/"

        self.robots_parsers: Dict[str, RobotFileParser] = {}
        self.xml_root = None
        self.xml_doc = None
        self.session = None
        self.pbar = None
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        
    async def init_session(self):
        """Initialize aiohttp session."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }
        self.session = aiohttp.ClientSession(headers=headers)
        
    async def close_session(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            
    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if not self.respect_robots:
            return True
            
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        if domain not in self.robots_parsers:
            # Initialize and fetch robots.txt
            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            parser = RobotFileParser()
            parser.set_url(robots_url)
            
            try:
                parser.read()
                self.robots_parsers[domain] = parser
            except Exception as e:
                logger.warning(f"Could not fetch robots.txt for {domain}: {e}")
                # If we can't fetch robots.txt, assume OK to crawl
                return True
        
        return self.robots_parsers[domain].can_fetch(self.user_agent, url)
    
    def should_crawl(self, url: str) -> bool:
        """Determine if URL should be crawled based on patterns, domain, path, etc."""
        parsed_url = urlparse(url)
        
        # Only handle http/https
        if parsed_url.scheme not in ('http', 'https'):
            return False
            
        # If we don't follow external links, only crawl the same domain
        if not self.follow_links and parsed_url.netloc != self.domain:
            return False

        # If --restrict-path is set, require the path to start with start_url_path
        if self.restrict_path:
            if not parsed_url.path.startswith(self.start_url_path):
                return False

        # Skip if we've already visited
        if url in self.visited_urls:
            return False
            
        # Skip if we've hit max pages
        if self.pages_crawled >= self.max_pages:
            return False
            
        # If include_pattern is set, skip URLs that do not match it
        if self.include_pattern and not self.include_pattern.search(url):
            return False
            
        # If exclude_pattern is set, skip URLs that do match it
        if self.exclude_pattern and self.exclude_pattern.search(url):
            return False
            
        # Check robots.txt
        if not self.can_fetch(url):
            logger.debug(f"Skipping {url} due to robots.txt restrictions")
            return False
            
        return True
        
    async def fetch_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch URL content with error handling."""
        if not self.session:
            await self.init_session()
            
        try:
            async with self.session.get(url, timeout=self.timeout) as response:
                if response.status != 200:
                    error_msg = f"HTTP Error {response.status}: {response.reason}"
                    return None, error_msg
                    
                # Check content type
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('text/html'):
                    error_msg = f"Skipping non-HTML content: {content_type}"
                    return None, error_msg
                    
                html = await response.text()
                return html, None
                
        except asyncio.TimeoutError:
            return None, "Request timed out"
        except aiohttp.ClientError as e:
            return None, f"Client error: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from HTML content."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            # Skip anchors, js calls, etc.
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
                
            full_url = urljoin(base_url, href)
            full_url = full_url.split('#')[0]  # Remove fragments
            
            links.append(full_url)
            
        return links
        
    def detect_code_language(self, code: str) -> str:
        """Attempt to detect programming language of code block with naive heuristics."""
        if re.search(r'^\s*(import|from)\s+\w+\s+import|def\s+\w+\s*\(|class\s+\w+[:\(]', code):
            return "python"
        elif re.search(r'^\s*(function|const|let|var|import)\s+|=\>|{\s*\n|export\s+', code):
            return "javascript"
        elif re.search(r'^\s*(#include|int\s+main|using\s+namespace)', code):
            return "cpp"
        elif re.search(r'^\s*(public\s+class|import\s+java|@Override)', code):
            return "java"
        elif re.search(r'<\?php|\$\w+\s*=', code):
            return "php"
        elif re.search(r'^\s*(use\s+|fn\s+\w+|let\s+mut|impl)', code):
            return "rust"
        elif re.search(r'^\s*(package\s+main|import\s+\(|func\s+\w+\s*\()', code):
            return "go"
        elif re.search(r'<html|<body|<div|<script|<style', code):
            return "html"
        elif re.search(r'^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE TABLE)', code, re.IGNORECASE):
            return "sql"
        return "code"
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]', ' ', text)
        text = re.sub(r'[\u2018\u2019]', "'", text)
        text = re.sub(r'[\u201C\u201D]', '"', text)
        return text
        
    def process_html(self, html: str, url: str) -> Dict:
        """Process HTML content and extract structured information."""
        try:
            doc = Document(html)
            title = doc.title()
            
            if self.clean_html:
                main_content = doc.summary()
                soup = BeautifulSoup(main_content, 'lxml')
            else:
                soup = BeautifulSoup(html, 'lxml')
                
            if self.strip_js:
                for script in soup.find_all('script'):
                    script.decompose()
            if self.strip_css:
                for style in soup.find_all('style'):
                    style.decompose()
            if self.strip_comments:
                for comment in soup.find_all(string=lambda text: isinstance(text, bs4.Comment)):
                    comment.extract()
                    
            meta_tags = {}
            for meta in soup.find_all('meta'):
                if meta.get('name') and meta.get('content'):
                    meta_tags[meta['name']] = meta['content']
                elif meta.get('property') and meta.get('content'):
                    meta_tags[meta['property']] = meta['content']
            
            content_data = self.extract_structured_content(soup, url)
            
            return {
                'url': url,
                'title': title,
                'meta': meta_tags,
                'content': content_data
            }
            
        except Exception as e:
            logger.error(f"Error processing HTML for {url}: {e}")
            return {
                'url': url,
                'title': "Error processing page",
                'meta': {},
                'content': []
            }
            
    def extract_structured_content(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Extract structured content from the page."""
        content_blocks = []
        
        # Headings
        if self.extract_headings:
            content_blocks.extend(self.extract_hierarchical_content(soup))
        else:
            # Fallback: just paragraphs, lists, etc.
            for p in soup.find_all('p'):
                text = self.clean_text(p.get_text())
                if text:
                    content_blocks.append({
                        'type': 'paragraph',
                        'text': text
                    })
            for list_elem in soup.find_all(['ul', 'ol']):
                items = []
                for li in list_elem.find_all('li'):
                    item_text = self.clean_text(li.get_text())
                    if item_text:
                        items.append(item_text)
                if items:
                    content_blocks.append({
                        'type': 'list',
                        'list_type': 'ordered' if list_elem.name == 'ol' else 'unordered',
                        'items': items
                    })
        
        # Code blocks
        if self.include_code:
            for code_elem in soup.find_all(['pre', 'code']):
                code_text = code_elem.get_text()
                if code_text and code_text.strip():
                    lang = self.detect_code_language(code_text)
                    content_blocks.append({
                        'type': 'code',
                        'language': lang,
                        'code': code_text
                    })
        
        # Images
        if self.include_images:
            for img in soup.find_all('img'):
                alt_text = img.get('alt', '')
                src = img.get('src', '')
                if src:
                    img_url = urljoin(url, src)
                    content_blocks.append({
                        'type': 'image',
                        'url': img_url,
                        'alt_text': alt_text
                    })
        
        # Tables
        for table in soup.find_all('table'):
            table_data = []
            headers = []
            
            thead = table.find('thead')
            if thead:
                th_rows = thead.find_all('tr')
                for row in th_rows:
                    headers.extend([self.clean_text(cell.get_text()) for cell in row.find_all(['th', 'td'])])
            else:
                first_row = table.find('tr')
                if first_row:
                    th_cells = first_row.find_all('th')
                    if th_cells:
                        headers = [self.clean_text(cell.get_text()) for cell in th_cells]
            
            tbody = table.find('tbody') or table
            rows = tbody.find_all('tr')
            # If first row contains <th>, skip it since it's effectively the header
            start_idx = 1 if headers and not thead and table.find('tr').find('th') else 0
            
            for row in rows[start_idx:]:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [self.clean_text(cell.get_text()) for cell in cells]
                    table_data.append(row_data)
            
            if table_data:
                content_blocks.append({
                    'type': 'table',
                    'headers': headers,
                    'rows': table_data
                })
                
        return content_blocks
        
    def extract_hierarchical_content(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract content with hierarchical structure based on headings."""
        content_blocks = []
        
        elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol'])
        
        current_heading = {'type': 'heading', 'level': 0, 'text': '', 'children': []}
        heading_stack = [current_heading]
        
        for elem in elements:
            if elem.name.startswith('h') and len(elem.name) == 2:
                level = int(elem.name[1])
                heading_text = self.clean_text(elem.get_text())
                if not heading_text:
                    continue
                while level <= heading_stack[-1]['level'] and len(heading_stack) > 1:
                    heading_stack.pop()
                new_heading = {
                    'type': 'heading',
                    'level': level,
                    'text': heading_text,
                    'children': []
                }
                heading_stack[-1]['children'].append(new_heading)
                heading_stack.append(new_heading)
            else:
                if elem.name == 'p':
                    text = self.clean_text(elem.get_text())
                    if text:
                        heading_stack[-1]['children'].append({
                            'type': 'paragraph',
                            'text': text
                        })
                elif elem.name in ('ul', 'ol'):
                    items = []
                    for li in elem.find_all('li'):
                        item_text = self.clean_text(li.get_text())
                        if item_text:
                            items.append(item_text)
                    if items:
                        heading_stack[-1]['children'].append({
                            'type': 'list',
                            'list_type': 'ordered' if elem.name == 'ol' else 'unordered',
                            'items': items
                        })
        
        def flatten_hierarchy(node):
            result = []
            if node['type'] == 'heading' and node['level'] > 0:
                result.append({
                    'type': 'heading',
                    'level': node['level'],
                    'text': node['text']
                })
            for child in node.get('children', []):
                if isinstance(child, dict) and child.get('type') == 'heading':
                    result.extend(flatten_hierarchy(child))
                else:
                    result.append(child)
            return result
            
        for child in heading_stack[0]['children']:
            content_blocks.extend(flatten_hierarchy(child))
            
        return content_blocks
        
    def create_xml_document(self) -> None:
        """Create the XML document structure."""
        self.xml_doc = xml.dom.minidom.getDOMImplementation().createDocument(None, "documentation", None)
        self.xml_root = self.xml_doc.documentElement
        
        metadata = self.xml_doc.createElement("metadata")
        
        self.add_element(metadata, "created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.add_element(metadata, "crawler_version", "1.0")
        self.add_element(metadata, "start_url", self.start_url)
        self.add_element(metadata, "pages_crawled", str(self.pages_crawled))
        self.add_element(metadata, "max_depth", str(self.max_depth))
        
        self.xml_root.appendChild(metadata)
        
    def add_element(self, parent, name, text=None, attrs=None):
        """Helper method to add an XML element with optional text and attributes."""
        element = self.xml_doc.createElement(name)
        
        if text is not None:
            text_node = self.xml_doc.createTextNode(str(text))
            element.appendChild(text_node)
            
        if attrs:
            for key, value in attrs.items():
                element.setAttribute(key, str(value))
                
        parent.appendChild(element)
        return element
        
    def add_page_to_xml(self, page_data: Dict) -> None:
        """Add a page to the XML document."""
        page_elem = self.xml_doc.createElement("page")
        page_elem.setAttribute("url", page_data['url'])
        
        self.add_element(page_elem, "title", page_data['title'])
        
        if page_data['meta']:
            meta_elem = self.xml_doc.createElement("meta")
            for key, value in page_data['meta'].items():
                self.add_element(meta_elem, "meta_item", value, {"name": key})
            page_elem.appendChild(meta_elem)
            
        content_elem = self.xml_doc.createElement("content")
        for block in page_data['content']:
            block_type = block['type']
            if block_type == 'paragraph':
                self.add_element(content_elem, "paragraph", block['text'])
            elif block_type == 'heading':
                self.add_element(content_elem, "heading", block['text'], {"level": str(block['level'])})
            elif block_type == 'list':
                list_elem = self.xml_doc.createElement("list")
                list_elem.setAttribute("type", block['list_type'])
                for item in block['items']:
                    self.add_element(list_elem, "item", item)
                content_elem.appendChild(list_elem)
            elif block_type == 'code':
                self.add_element(content_elem, "code", block['code'], {"language": block['language']})
            elif block_type == 'image':
                self.add_element(content_elem, "image", block['alt_text'], {"src": block['url']})
            elif block_type == 'table':
                table_elem = self.xml_doc.createElement("table")
                if block['headers']:
                    headers_elem = self.xml_doc.createElement("headers")
                    for header in block['headers']:
                        self.add_element(headers_elem, "header", header)
                    table_elem.appendChild(headers_elem)
                for row in block['rows']:
                    row_elem = self.xml_doc.createElement("row")
                    for cell in row:
                        self.add_element(row_elem, "cell", cell)
                    table_elem.appendChild(row_elem)
                content_elem.appendChild(table_elem)
        page_elem.appendChild(content_elem)
        self.xml_root.appendChild(page_elem)
        
    def save_xml_to_file(self) -> None:
        """Save the XML document to file."""
        if not self.xml_doc:
            logger.error("No XML doc to save.")
            return

        pretty_xml = self.xml_doc.toprettyxml(indent="  ")
        # Remove empty lines from toprettyxml
        pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            logger.info(f"XML saved to {self.output_file}")
            print(f"{Fore.GREEN}Successfully saved documentation to {self.output_file}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Error saving XML file: {e}")
            print(f"{Fore.RED}Error saving XML file: {e}{Style.RESET_ALL}")
            return
        
        try:
            import pyperclip
            pyperclip.copy(pretty_xml)
            logger.info("Copied XML to clipboard")
            print(f"{Fore.YELLOW}XML output copied to clipboard.{Style.RESET_ALL}")
        except ImportError:
            logger.warning("pyperclip not installed; cannot copy to clipboard")
        
        # Attempt to compute token count if tiktoken is installed
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(pretty_xml)
            token_count = len(tokens)
            print(f"{Fore.MAGENTA}Total tokens in output: {token_count}{Style.RESET_ALL}")
        except ImportError:
            logger.debug("tiktoken not installed; skipping token count.")
        
        # Final stats
        print(f"\n{Fore.CYAN}Crawl Statistics:{Style.RESET_ALL}")
        print(f"Pages crawled: {self.pages_crawled}")
        print(f"Failed URLs: {len(self.failed_urls)}")
        if self.failed_urls and self.verbose:
            print(f"\n{Fore.YELLOW}Failed URLs (showing up to 10):{Style.RESET_ALL}")
            for url, error in self.failed_urls[:10]:
                print(f"- {url}: {error}")
            if len(self.failed_urls) > 10:
                print(f"... and {len(self.failed_urls) - 10} more. Check logs for details.")

    async def worker(self, queue: "asyncio.Queue[Tuple[str,int]]"):
        """Async worker that crawls pages from the queue."""
        while True:
            try:
                url, depth = await queue.get()
                
                if self.pages_crawled >= self.max_pages:
                    queue.task_done()
                    continue
                
                if not self.should_crawl(url):
                    queue.task_done()
                    continue

                self.visited_urls.add(url)
                await asyncio.sleep(self.delay)
                logger.debug(f"Fetching {url} at depth {depth}")
                
                html, error = await self.fetch_url(url)
                if error:
                    logger.warning(f"Failed to fetch {url}: {error}")
                    self.failed_urls.append((url, error))
                    queue.task_done()
                    continue

                page_data = self.process_html(html, url)
                self.add_page_to_xml(page_data)

                self.pages_crawled += 1
                if self.pbar:
                    self.pbar.update(1)
                
                if depth < self.max_depth:
                    soup = BeautifulSoup(html, 'lxml')
                    links = self.extract_links(soup, url)
                    for link in links:
                        if self.pages_crawled < self.max_pages:
                            await queue.put((link, depth + 1))

                queue.task_done()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                queue.task_done()

    async def crawl(self) -> None:
        """Main crawl function with concurrency."""
        try:
            await self.init_session()
            self.create_xml_document()

            queue: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
            await queue.put((self.start_url, 0))
            self.pbar = tqdm.tqdm(total=self.max_pages, desc="Crawling", unit="page")

            tasks = []
            for _ in range(self.concurrency):
                t = asyncio.create_task(self.worker(queue))
                tasks.append(t)

            await queue.join()

            for t in tasks:
                t.cancel()

            self.pbar.close()
            self.save_xml_to_file()

        except KeyboardInterrupt:
            logger.info("Crawl interrupted by user")
            print(f"\n{Fore.YELLOW}Crawl interrupted. Saving progress...{Style.RESET_ALL}")
            if self.xml_root and self.pages_crawled > 0:
                self.save_xml_to_file()
        except Exception as e:
            logger.error(f"Crawl error: {e}")
            print(f"{Fore.RED}Error during crawl: {e}{Style.RESET_ALL}")
        finally:
            if self.pbar:
                self.pbar.close()
            await self.close_session()

        # After crawling, optionally chunk+embed
        if self.chunk_and_embed:
            embedder = Embedder(
                embedding_function_name=self.embedding_function_name,
                openai_api_key=self.openai_api_key,
                cohere_api_key=self.cohere_api_key,
                huggingface_model_name=self.huggingface_model_name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                chunk_by_chars=self.chunk_by_chars,
                ignore_content_boundaries=self.ignore_content_boundaries,
                chroma_collection=self.chroma_collection,
                chroma_persist_dir=self.chroma_persist_dir,
                output_chunks=self.output_chunks,
            )
            embedder.run_embedding_workflow(self.output_file)


###############################################################################
#                           EMBEDDING / CHUNKING CODE                         #
###############################################################################

class Embedder:
    """Handles chunking of XML content and embedding with chosen embedding provider, storing results in ChromaDB."""

    def __init__(
        self,
        embedding_function_name: str = "default",
        openai_api_key: Optional[str] = None,
        cohere_api_key: Optional[str] = None,
        huggingface_model_name: Optional[str] = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        chunk_by_chars: bool = False,
        ignore_content_boundaries: bool = False,
        chroma_collection: str = "documentation",
        chroma_persist_dir: str = "./chroma_db",
        output_chunks: Optional[str] = None,
    ):
        self.embedding_function_name = embedding_function_name
        self.openai_api_key = openai_api_key
        self.cohere_api_key = cohere_api_key
        self.huggingface_model_name = huggingface_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_by_chars = chunk_by_chars
        self.ignore_content_boundaries = ignore_content_boundaries
        self.chroma_collection = chroma_collection
        self.chroma_persist_dir = chroma_persist_dir
        self.output_chunks = output_chunks

        self.embedding_fn = self._get_embedding_fn()

    def _get_embedding_fn(self):
        """Return an embedding function object recognized by Chroma."""
        import chromadb
        from chromadb.utils import embedding_functions

        # Default embedding
        if self.embedding_function_name == "default":
            return embedding_functions.DefaultEmbeddingFunction()

        if self.embedding_function_name == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("You must `pip install openai` to use OpenAI embeddings.")
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not provided. Use --openai-api-key=<key>.")
            openai.api_key = self.openai_api_key

            # Chroma has a helper for this
            return embedding_functions.OpenAIEmbeddingFunction(api_key=self.openai_api_key, model_name="text-embedding-ada-002")

        if self.embedding_function_name == "cohere":
            try:
                import cohere
            except ImportError:
                raise ImportError("You must `pip install cohere` to use Cohere embeddings.")
            if not self.cohere_api_key:
                raise ValueError("Cohere API key not provided. Use --cohere-api-key=<key>.")
            return embedding_functions.CohereEmbeddingFunction(api_key=self.cohere_api_key)

        if self.embedding_function_name == "huggingface":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("You must `pip install sentence-transformers` to use huggingface embeddings.")
            model_name = self.huggingface_model_name or "sentence-transformers/all-MiniLM-L6-v2"
            return embedding_functions.HuggingFaceEmbeddingFunction(model_name=model_name)

        # Fall back to default if none matched
        return embedding_functions.DefaultEmbeddingFunction()

    def run_embedding_workflow(self, xml_file: str):
        """Main entry: parse XML, chunk it, embed in ChromaDB, optionally save chunk JSON."""
        print(f"{Fore.CYAN}Embedding workflow started...{Style.RESET_ALL}")
        # Parse XML
        docs = self._load_and_flatten_xml(xml_file)

        # Chunk
        chunks = self._chunk_documents(docs)

        # Save chunks if requested
        if self.output_chunks:
            import json
            with open(self.output_chunks, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2)
            print(f"{Fore.GREEN}Chunked output saved to {self.output_chunks}{Style.RESET_ALL}")

        # Store in Chroma
        self._store_in_chroma(chunks)
        print(f"{Fore.GREEN}Embedding workflow complete! Stored {len(chunks)} chunks in ChromaDB.{Style.RESET_ALL}")

    def _load_and_flatten_xml(self, xml_file: str) -> List[Dict]:
        """Load the XML file and flatten each page's content blocks into text segments."""
        from xml.dom.minidom import parse

        dom = parse(xml_file)
        pages = dom.getElementsByTagName("page")
        docs = []

        for page in pages:
            url = page.getAttribute("url")
            title_nodes = page.getElementsByTagName("title")
            title = title_nodes[0].childNodes[0].nodeValue if title_nodes and title_nodes[0].childNodes else ""

            # Flatten each page's text content
            content_nodes = page.getElementsByTagName("content")
            texts = []
            if content_nodes:
                # For each child element (paragraph, heading, list->item, etc.)
                for child in content_nodes[0].childNodes:
                    if not child.nodeName or child.nodeName == "#text":
                        continue
                    if child.nodeName in ["paragraph", "heading", "code", "image"]:
                        node_text = (child.childNodes[0].nodeValue if child.childNodes else "").strip()
                        if node_text:
                            texts.append(node_text)
                    elif child.nodeName == "list":
                        for item in child.getElementsByTagName("item"):
                            if item.childNodes:
                                item_text = item.childNodes[0].nodeValue.strip()
                                if item_text:
                                    texts.append(item_text)
                    elif child.nodeName == "table":
                        # Optional: flatten table into lines
                        # For example, each row => a single line
                        row_nodes = child.getElementsByTagName("row")
                        for row_node in row_nodes:
                            cell_texts = []
                            for cell in row_node.getElementsByTagName("cell"):
                                if cell.childNodes:
                                    cell_texts.append(cell.childNodes[0].nodeValue.strip())
                            if cell_texts:
                                texts.append(" | ".join(cell_texts))

            combined_text = f"{title}\n\n" + "\n\n".join(texts)
            docs.append({"url": url, "text": combined_text.strip()})

        return docs

    def _chunk_documents(self, docs: List[Dict]) -> List[Dict]:
        """Split documents into chunks based on token or character length."""
        # Try to import tiktoken
        if not self.chunk_by_chars:
            try:
                import tiktoken
                encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                print(f"{Fore.YELLOW}tiktoken not installed; falling back to character-based chunking.{Style.RESET_ALL}")
                self.chunk_by_chars = True
                encoder = None
        else:
            encoder = None

        all_chunks = []
        for doc in docs:
            text = doc["text"]
            if not text:
                continue
            if self.chunk_by_chars:
                # Character-based chunking
                doc_chunks = self._split_by_chars(text, self.chunk_size, self.chunk_overlap)
            else:
                # Token-based chunking
                doc_chunks = self._split_by_tokens(text, encoder, self.chunk_size, self.chunk_overlap)
            for chunk_id, chunk_text in enumerate(doc_chunks):
                all_chunks.append({
                    "id": f"{doc['url']}-chunk-{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        "source_url": doc['url']
                    }
                })
        return all_chunks

    def _split_by_chars(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Chunk a string by character count, optionally respecting boundaries if requested."""
        if self.ignore_content_boundaries:
            # Just slice by fixed windows
            return self._sliding_window(text, chunk_size, overlap)
        else:
            # Attempt to split on paragraphs or lines, then recombine up to chunk_size
            paragraphs = text.split("\n\n")
            chunks = []
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) <= chunk_size:
                    current_chunk += (para + "\n\n")
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Now handle overlap if requested
            if overlap > 0:
                # We'll do a simple approach: in character-based, we re-append the last `overlap` chars of the last chunk
                # to the next chunk. This is naive but workable.
                final_chunks = []
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        final_chunks.append(chunk)
                        continue
                    # Prepend overlap from previous chunk
                    prev_chunk = final_chunks[-1]
                    overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
                    final_chunks.append(overlap_text + " " + chunk)
                return final_chunks
            else:
                return chunks

    def _split_by_tokens(self, text: str, encoder, chunk_size: int, overlap: int) -> List[str]:
        """Token-based chunking with optional content-boundary respect."""
        if not encoder:
            # Fallback to char-based
            return self._split_by_chars(text, chunk_size, overlap)

        tokens = encoder.encode(text)
        if self.ignore_content_boundaries:
            return self._split_tokens_sliding_window(tokens, encoder, chunk_size, overlap)
        else:
            # We'll do a paragraph-based approach, but measure each paragraph in tokens
            paragraphs = text.split("\n\n")
            chunked = []
            current_tokens = []

            for para in paragraphs:
                para_tokens = encoder.encode(para)
                if len(current_tokens) + len(para_tokens) <= chunk_size:
                    current_tokens.extend(para_tokens + encoder.encode("\n\n"))
                else:
                    if current_tokens:
                        chunked.append(encoder.decode(current_tokens).strip())
                    current_tokens = para_tokens + encoder.encode("\n\n")

            if current_tokens:
                chunked.append(encoder.decode(current_tokens).strip())

            # Overlap
            if overlap > 0:
                final_chunks = []
                for i, chunk in enumerate(chunked):
                    if i == 0:
                        final_chunks.append(chunk)
                        continue
                    prev_chunk = final_chunks[-1]
                    prev_tokens = encoder.encode(prev_chunk)
                    overlap_tokens = prev_tokens[-overlap:] if len(prev_tokens) > overlap else prev_tokens
                    # Combine
                    final_chunks.append(encoder.decode(overlap_tokens) + " " + chunk)
                return final_chunks
            else:
                return chunked

    def _sliding_window(self, text: str, size: int, overlap: int) -> List[str]:
        """Return a list of fixed-size segments from text, sliding with overlap."""
        segments = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + size
            segment = text[start:end]
            segments.append(segment)
            start += (size - overlap)  # slide
            if start >= text_length:
                break
        return segments

    def _split_tokens_sliding_window(self, tokens: List[int], encoder, size: int, overlap: int) -> List[str]:
        """Fixed-size token windows with overlap."""
        results = []
        start = 0
        total_tokens = len(tokens)
        while start < total_tokens:
            end = start + size
            segment_tokens = tokens[start:end]
            segment_text = encoder.decode(segment_tokens)
            results.append(segment_text)
            start += (size - overlap)
            if start >= total_tokens:
                break
        return results

    def _store_in_chroma(self, chunks: List[Dict]):
        """Store chunk texts and embeddings in ChromaDB."""
        import chromadb
        from chromadb.config import Settings

        client = chromadb.Client(
            Settings(
                persist_directory=self.chroma_persist_dir,
                anonymized_telemetry=False
            )
        )
        # Try to get collection; if not exist, create it
        collection = None
        try:
            collection = client.get_collection(self.chroma_collection, embedding_function=self.embedding_fn)
        except ValueError:
            # Does not exist
            collection = client.create_collection(self.chroma_collection, embedding_function=self.embedding_fn)

        # We add in small batches to avoid issues with large inserts
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            collection.add(
                documents=[ch['text'] for ch in batch],
                ids=[ch['id'] for ch in batch],
                metadatas=[ch['metadata'] for ch in batch]
            )


###############################################################################
#                          CLI ARGUMENT PARSING & MAIN                        #
###############################################################################
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="docs2xml: Crawl documentation to XML, optionally chunk & embed in ChromaDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ---------------------------
    # crawl subcommand
    # ---------------------------
    crawl_parser = subparsers.add_parser("crawl", help="Crawl a website and output XML")
    crawl_parser.add_argument("start_url", help="Starting URL to crawl")
    crawl_parser.add_argument("--output-file", default="docs_archive.xml", help="Output XML file")
    crawl_parser.add_argument("--max-depth", type=int, default=5, help="Maximum crawl depth")
    crawl_parser.add_argument("--max-pages", type=int, default=1000, help="Maximum pages to crawl")
    crawl_parser.add_argument("--user-agent", default="DocCrawler/1.0", help="User agent string")
    crawl_parser.add_argument("--delay", type=float, default=0.2, help="Delay between requests in seconds")
    crawl_parser.add_argument("--include-pattern", default="None", help="URL pattern to include (regex)")
    crawl_parser.add_argument("--exclude-pattern", default="None", help="URL pattern to exclude (regex)")
    crawl_parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    crawl_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    crawl_parser.add_argument("--include-images", action="store_true", help="Include image descriptions in output")
    crawl_parser.add_argument("--include-code", action="store_true", default=True, help="Include code blocks with language detection")
    crawl_parser.add_argument("--extract-headings", action="store_true", default=True, help="Extract headings hierarchically")
    crawl_parser.add_argument("--follow-links", action="store_true", help="Follow links to external domains")
    crawl_parser.add_argument("--clean-html", action="store_true", default=True, help="Use readability-lxml to focus on main content")
    crawl_parser.add_argument("--strip-js", action="store_true", default=True, help="Remove <script> tags")
    crawl_parser.add_argument("--strip-css", action="store_true", default=True, help="Remove <style> tags")
    crawl_parser.add_argument("--strip-comments", action="store_true", default=True, help="Remove HTML comments")
    crawl_parser.add_argument("--robots-txt", action="store_true", default=False, help="Respect robots.txt rules")
    crawl_parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests")
    crawl_parser.add_argument("--restrict-path", action="store_true", default=False,
                              help="Restrict crawling to paths starting with the start_url's path")

    # chunk+embed add-ons
    crawl_parser.add_argument("--chunk-and-embed", action="store_true", default=False,
                              help="After crawling, chunk and embed the resulting XML into ChromaDB")
    crawl_parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size (tokens or chars)")
    crawl_parser.add_argument("--chunk-overlap", type=int, default=128, help="Overlap (tokens or chars) between chunks")
    crawl_parser.add_argument("--chunk-by-chars", action="store_true", default=False,
                              help="Chunk by characters instead of tokens (default uses tiktoken for tokens)")
    crawl_parser.add_argument("--ignore-content-boundaries", action="store_true", default=False,
                              help="Ignore paragraph/heading boundaries and chunk purely by size+overlap")
    crawl_parser.add_argument("--output-chunks", default=None,
                              help="Optional JSON file to save chunks before embedding")

    # embedding function arguments
    crawl_parser.add_argument("--embedding-function", default="default", choices=["default", "openai", "cohere", "huggingface"],
                              help="Select which embedding function to use")
    crawl_parser.add_argument("--openai-api-key", default=None, help="OpenAI API key for openai embeddings")
    crawl_parser.add_argument("--cohere-api-key", default=None, help="Cohere API key for cohere embeddings")
    crawl_parser.add_argument("--huggingface-model-name", default=None,
                              help="HuggingFace model name (e.g. 'sentence-transformers/all-MiniLM-L6-v2')")
    crawl_parser.add_argument("--chroma-collection", default="documentation", help="Name of the ChromaDB collection to use")
    crawl_parser.add_argument("--chroma-persist-dir", default="./chroma_db", help="Directory for ChromaDB persistence")


    # ---------------------------
    # embed subcommand
    # ---------------------------
    embed_parser = subparsers.add_parser("embed", help="Embed an existing XML file into ChromaDB")
    embed_parser.add_argument("xml_file", help="Path to the XML file previously created by docs2xml")
    embed_parser.add_argument("--chunk-size", type=int, default=1024, help="Chunk size (tokens or chars)")
    embed_parser.add_argument("--chunk-overlap", type=int, default=128, help="Overlap (tokens or chars) between chunks")
    embed_parser.add_argument("--chunk-by-chars", action="store_true", default=False,
                              help="Chunk by characters instead of tokens (default uses tiktoken for tokens)")
    embed_parser.add_argument("--ignore-content-boundaries", action="store_true", default=False,
                              help="Ignore paragraph/heading boundaries and chunk purely by size+overlap")
    embed_parser.add_argument("--output-chunks", default=None,
                              help="Optional JSON file to save chunks before embedding")

    embed_parser.add_argument("--embedding-function", default="default", choices=["default", "openai", "cohere", "huggingface"],
                              help="Select which embedding function to use")
    embed_parser.add_argument("--openai-api-key", default=None, help="OpenAI API key for openai embeddings")
    embed_parser.add_argument("--cohere-api-key", default=None, help="Cohere API key for cohere embeddings")
    embed_parser.add_argument("--huggingface-model-name", default=None,
                              help="HuggingFace model name (e.g. 'sentence-transformers/all-MiniLM-L6-v2')")
    embed_parser.add_argument("--chroma-collection", default="documentation", help="Name of the ChromaDB collection to use")
    embed_parser.add_argument("--chroma-persist-dir", default="./chroma_db", help="Directory for ChromaDB persistence")


    # ---------------------------
    # version subcommand
    # ---------------------------
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    return args


async def main():
    args = parse_arguments()
    
    if args.command == "version":
        print("docs2xml version 2.0 (with chunk+embed)")
        return

    if args.command == "crawl":
        banner = f"""
{Fore.CYAN}
 ____              ____                    _           
|  _ \\  ___   ___ / ___|_ __ __ ___      _| | ___ _ __ 
| | | |/ _ \\ / __| |   | '__/ _` \\ \\ /\\ / / |/ _ \\ '__|
| |_| | (_) | (__| |___| | | (_| |\\ V  V /| |  __/ |   
|____/ \\___/ \\___|\\____|_|  \\__,_| \\_/\\_/ |_|\\___|_|   
{Style.RESET_ALL}
docs2xml v2.0: Crawl site -> XML, optionally chunk+embed in ChromaDB.
Starting crawl from: {Fore.GREEN}{args.start_url}{Style.RESET_ALL}
"""
        print(banner)
        
        crawler = DocCrawler(args)
        await crawler.crawl()

    elif args.command == "embed":
        # direct embed of existing XML
        embedder = Embedder(
            embedding_function_name=args.embedding_function,
            openai_api_key=args.openai_api_key,
            cohere_api_key=args.cohere_api_key,
            huggingface_model_name=args.huggingface_model_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            chunk_by_chars=args.chunk_by_chars,
            ignore_content_boundaries=args.ignore_content_boundaries,
            chroma_collection=args.chroma_collection,
            chroma_persist_dir=args.chroma_persist_dir,
            output_chunks=args.output_chunks,
        )
        embedder.run_embedding_workflow(args.xml_file)


if __name__ == "__main__":
    # For Windows event loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
