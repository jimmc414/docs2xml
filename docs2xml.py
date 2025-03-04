#!/usr/bin/env python3
"""
DocCrawler: A CLI utility to crawl website documentation and archive it in XML format for LLM ingestion.

Usage:
    python doc_crawler.py crawl <start_url> [options]
    python doc_crawler.py --help

Options:
    --output-file=<file>        Output XML file [default: docs_archive.xml]
    --max-depth=<depth>         Maximum crawl depth [default: 5]
    --max-pages=<pages>         Maximum pages to crawl [default: 1000]
    --user-agent=<agent>        User agent string [default: DocCrawler/1.0]
    --delay=<seconds>           Delay between requests in seconds [default: 0.2]
    --include-pattern=<regex>   URL pattern to include [default: None]
    --exclude-pattern=<regex>   URL pattern to exclude [default: None]
    --timeout=<seconds>         Request timeout in seconds [default: 30]
    --verbose                   Verbose output [default: False]
    --include-images            Include image descriptions in output [default: False]
    --include-code              Include code blocks with language detection [default: True]
    --extract-headings          Extract and hierarchically organize headings [default: True]
    --follow-links              Follow links to external domains [default: False]
    --clean-html                Enhance cleaning of HTML content [default: True]
    --strip-js                  Remove JavaScript content [default: True]
    --strip-css                 Remove CSS content [default: True]
    --strip-comments            Remove HTML comments [default: True]
    --robots-txt                Respect robots.txt rules [default: False]
    --concurrency=<N>           Number of concurrent requests [default: 5]
"""

import sys

# Insert default subcommand "crawl" if none is provided
if len(sys.argv) <= 1 or sys.argv[1] not in ['crawl', 'version']:
    sys.argv.insert(1, 'crawl')

import argparse
import asyncio
import logging
import re
import time
import xml.dom.minidom
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

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
    print("pip install aiohttp beautifulsoup4 lxml readability-lxml langdetect tqdm colorama python-robots")
    sys.exit(1)
#!/usr/bin/env python3
"""
DocCrawler: A CLI utility to crawl website documentation and archive it in XML format for LLM ingestion.

Usage:
    python doc_crawler.py crawl <start_url> [options]
    python doc_crawler.py --help

Options:
    --output-file=<file>        Output XML file [default: docs_archive.xml]
    --max-depth=<depth>         Maximum crawl depth [default: 5]
    --max-pages=<pages>         Maximum pages to crawl [default: 1000]
    --user-agent=<agent>        User agent string [default: DocCrawler/1.0]
    --delay=<seconds>           Delay between requests in seconds [default: 0.2]
    --include-pattern=<regex>   URL pattern to include [default: None]
    --exclude-pattern=<regex>   URL pattern to exclude [default: None]
    --timeout=<seconds>         Request timeout in seconds [default: 30]
    --verbose                   Verbose output [default: False]
    --include-images            Include image descriptions in output [default: False]
    --include-code              Include code blocks with language detection [default: True]
    --extract-headings          Extract and hierarchically organize headings [default: True]
    --follow-links              Follow links to external domains [default: False]
    --clean-html                Enhance cleaning of HTML content [default: True]
    --strip-js                  Remove JavaScript content [default: True]
    --strip-css                 Remove CSS content [default: True]
    --strip-comments            Remove HTML comments [default: True]
    --robots-txt                Respect robots.txt rules [default: False]
    --concurrency=<N>           Number of concurrent requests [default: 5]
"""

import sys

# Insert default subcommand "crawl" if none is provided
if len(sys.argv) <= 1 or sys.argv[1] not in ['crawl', 'version']:
    sys.argv.insert(1, 'crawl')

import argparse
import asyncio
import logging
import re
import time
import xml.dom.minidom
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

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
    print("pip install aiohttp beautifulsoup4 lxml readability-lxml langdetect tqdm colorama python-robots")
    sys.exit(1)


# Initialize colorama for cross-platform colored output
colorama.init()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("doc_crawler")

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
        
        # NEW: concurrency argument
        self.concurrency = args.concurrency
        
        # Initialize state
        self.visited_urls: Set[str] = set()
        # We'll now feed URLs into an asyncio.Queue, instead of storing them in a list.
        self.pages_crawled = 0
        self.failed_urls: List[Tuple[str, str]] = []  # (url, error_message)
        self.domain = urlparse(self.start_url).netloc
        self.robots_parsers: Dict[str, RobotFileParser] = {}
        self.xml_root = None
        self.session = None
        
        # Set up progress bar (will be used in concurrency)
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
                # Synchronous fetch of robots.txt - only happens once per domain
                parser.read()
                self.robots_parsers[domain] = parser
                
                if not any(rule.pattern for rule in parser._rules):  # If there are no Disallow directives
                    return True  # Allow crawling by default
            except Exception as e:
                logger.warning(f"Could not fetch robots.txt for {domain}: {e}")
                # If we can't fetch robots.txt, we'll assume it's okay to crawl
                return True
        
        return self.robots_parsers[domain].can_fetch(self.user_agent, url)
    
    def should_crawl(self, url: str) -> bool:
        """Determine if URL should be crawled based on patterns and domain."""
        parsed_url = urlparse(url)
        
        # Skip URLs that are not http or https
        if parsed_url.scheme not in ('http', 'https'):
            return False
            
        # Skip URLs that have already been visited
        if url in self.visited_urls:
            return False
            
        # Skip URLs based on max pages
        if self.pages_crawled >= self.max_pages:
            return False
            
        # Skip URLs that don't match include pattern
        if self.include_pattern and not self.include_pattern.search(url):
            return False
            
        # Skip URLs that match exclude pattern
        if self.exclude_pattern and self.exclude_pattern.search(url):
            return False
            
        # Handle external domains
        if not self.follow_links and parsed_url.netloc != self.domain:
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
            
            # Skip empty links, anchors, or javascript
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
                
            # Convert relative URL to absolute
            full_url = urljoin(base_url, href)
            
            # Remove fragments
            full_url = full_url.split('#')[0]
            
            links.append(full_url)
            
        return links
        
    def detect_code_language(self, code: str) -> str:
        """Attempt to detect programming language of code block."""
        # Simple heuristics for language detection
        if re.search(r'^\s*(import|from)\s+\w+\s+import|def\s+\w+\s*\(|class\s+\w+[:\(]', code):
            return "python"
        elif re.search(r'^\s*(function|const|let|var|import)\s+|=>|{\s*\n|export\s+', code):
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
        # Default to "code" if no specific language detected
        return "code"
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
            
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading and trailing whitespace
        text = text.strip()
        # Replace special Unicode whitespace characters
        text = re.sub(r'[\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]', ' ', text)
        # Replace special quotes with standard quotes
        text = re.sub(r'[\u2018\u2019]', "'", text)
        text = re.sub(r'[\u201C\u201D]', '"', text)
        
        return text
        
    def process_html(self, html: str, url: str) -> Dict:
        """Process HTML content and extract structured information."""
        # Use readability to extract main content
        try:
            doc = Document(html)
            title = doc.title()
            
            if self.clean_html:
                main_content = doc.summary()
                soup = BeautifulSoup(main_content, 'lxml')
            else:
                soup = BeautifulSoup(html, 'lxml')
                
            # Remove unwanted elements
            if self.strip_js:
                for script in soup.find_all('script'):
                    script.decompose()
                    
            if self.strip_css:
                for style in soup.find_all('style'):
                    style.decompose()
            
            # Remove HTML comments if strip_comments is True
            if self.strip_comments:
                for comment in soup.find_all(string=lambda text: isinstance(text, bs4.Comment)):
                    comment.extract()
                    
            # Extract page metadata
            meta_tags = {}
            for meta in soup.find_all('meta'):
                if meta.get('name') and meta.get('content'):
                    meta_tags[meta['name']] = meta['content']
                elif meta.get('property') and meta.get('content'):
                    meta_tags[meta['property']] = meta['content']
            
            # Extract structured content
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
        
        # Handle heading extraction if enabled
        if self.extract_headings:
            content_blocks.extend(self.extract_hierarchical_content(soup))
        else:
            # Extract paragraphs
            for p in soup.find_all('p'):
                text = self.clean_text(p.get_text())
                if text:
                    content_blocks.append({
                        'type': 'paragraph',
                        'text': text
                    })
                    
            # Extract lists
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
        
        # Extract code blocks
        if self.include_code:
            for code_elem in soup.find_all(['pre', 'code']):
                code_text = code_elem.get_text()
                if code_text and len(code_text.strip()) > 0:
                    lang = self.detect_code_language(code_text)
                    content_blocks.append({
                        'type': 'code',
                        'language': lang,
                        'code': code_text
                    })
        
        # Extract images if enabled
        if self.include_images:
            for img in soup.find_all('img'):
                alt_text = img.get('alt', '')
                src = img.get('src', '')
                if src:
                    # Convert relative URL to absolute
                    img_url = urljoin(url, src)
                    content_blocks.append({
                        'type': 'image',
                        'url': img_url,
                        'alt_text': alt_text
                    })
        
        # Extract tables
        for table in soup.find_all('table'):
            table_data = []
            headers = []
            
            # Extract headers first
            thead = table.find('thead')
            if thead:
                th_rows = thead.find_all('tr')
                for row in th_rows:
                    headers.extend([self.clean_text(cell.get_text()) for cell in row.find_all(['th', 'td'])])
            else:
                # If no thead, check first row for th tags
                first_row = table.find('tr')
                if first_row:
                    th_cells = first_row.find_all('th')
                    if th_cells:
                        headers = [self.clean_text(cell.get_text()) for cell in th_cells]
            
            # Extract rows
            tbody = table.find('tbody') or table
            rows = tbody.find_all('tr')
            
            # Skip first row if we already used it for headers
            start_idx = 1 if headers and not table.find('thead') and table.find('tr').find('th') else 0
            
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
        
        # Find all heading elements and content elements
        elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol'])
        
        current_heading = {'type': 'heading', 'level': 0, 'text': '', 'children': []}
        heading_stack = [current_heading]
        
        for elem in elements:
            if elem.name.startswith('h') and len(elem.name) == 2:
                # This is a heading element
                level = int(elem.name[1])
                heading_text = self.clean_text(elem.get_text())
                
                if not heading_text:
                    continue
                    
                # Pop stack until we find a heading of higher level (smaller number)
                while level <= heading_stack[-1]['level'] and len(heading_stack) > 1:
                    heading_stack.pop()
                
                # Create new heading at this level
                new_heading = {
                    'type': 'heading',
                    'level': level,
                    'text': heading_text,
                    'children': []
                }
                
                # Add to parent's children and update stack
                heading_stack[-1]['children'].append(new_heading)
                heading_stack.append(new_heading)
                
            else:
                # This is content under the current heading
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
        
        # Flatten the hierarchical structure
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
            
        # Process all top-level headings
        for child in heading_stack[0]['children']:
            content_blocks.extend(flatten_hierarchy(child))
            
        return content_blocks
        
    def create_xml_document(self) -> None:
        """Create the XML document structure."""
        # Create root element
        self.xml_doc = xml.dom.minidom.getDOMImplementation().createDocument(None, "documentation", None)
        self.xml_root = self.xml_doc.documentElement
        
        # Add crawler metadata
        metadata = self.xml_doc.createElement("metadata")
        
        # Add various metadata elements
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
        
        # Add title
        self.add_element(page_elem, "title", page_data['title'])
        
        # Add metadata
        if page_data['meta']:
            meta_elem = self.xml_doc.createElement("meta")
            for key, value in page_data['meta'].items():
                self.add_element(meta_elem, "meta_item", value, {"name": key})
            page_elem.appendChild(meta_elem)
            
        # Add content
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
                
                # Add headers if present
                if block['headers']:
                    headers_elem = self.xml_doc.createElement("headers")
                    for header in block['headers']:
                        self.add_element(headers_elem, "header", header)
                    table_elem.appendChild(headers_elem)
                    
                # Add rows
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
        # Format and indent the XML
        pretty_xml = self.xml_doc.toprettyxml(indent="  ")
        
        # Remove empty lines (a common issue with toprettyxml)
        pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
        
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
                
            logger.info(f"XML saved to {self.output_file}")
            print(f"{Fore.GREEN}Successfully saved documentation to {self.output_file}{Style.RESET_ALL}")
            
            # Print stats
            print(f"\n{Fore.CYAN}Crawl Statistics:{Style.RESET_ALL}")
            print(f"Pages crawled: {self.pages_crawled}")
            print(f"Failed URLs: {len(self.failed_urls)}")
            
            if self.failed_urls and self.verbose:
                print(f"\n{Fore.YELLOW}Failed URLs:{Style.RESET_ALL}")
                for url, error in self.failed_urls[:10]:  # Show only first 10 failures
                    print(f"- {url}: {error}")
                    
                if len(self.failed_urls) > 10:
                    print(f"... and {len(self.failed_urls) - 10} more. Check logs for details.")
                    
        except Exception as e:
            logger.error(f"Error saving XML file: {e}")
            print(f"{Fore.RED}Error saving XML file: {e}{Style.RESET_ALL}")
            return
        
        # Copy to clipboard (if pyperclip installed)
        try:
            import pyperclip
            pyperclip.copy(pretty_xml)
            logger.info("Copied XML to clipboard")
            print(f"{Fore.YELLOW}XML output copied to clipboard.{Style.RESET_ALL}")
        except ImportError:
            logger.warning("pyperclip not installed; cannot copy to clipboard")
        
        # Compute and print total token count (if tiktoken installed)
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")  # Commonly used for ChatGPT
            tokens = enc.encode(pretty_xml)
            token_count = len(tokens)
            print(f"{Fore.MAGENTA}Total tokens in output: {token_count}{Style.RESET_ALL}")
        except ImportError:
            logger.warning("tiktoken not installed; cannot compute token count")
        
        # Print crawl stats
        print(f"\n{Fore.CYAN}Crawl Statistics:{Style.RESET_ALL}")
        print(f"Pages crawled: {self.pages_crawled}")
        print(f"Failed URLs: {len(self.failed_urls)}")
        if self.failed_urls and self.verbose:
            print(f"\n{Fore.YELLOW}Failed URLs:{Style.RESET_ALL}")
            for url, error in self.failed_urls[:10]:  # only show first 10
                print(f"- {url}: {error}")
            if len(self.failed_urls) > 10:
                print(f"... and {len(self.failed_urls) - 10} more. Check logs for details.")

    async def worker(self, queue: "asyncio.Queue[Tuple[str,int]]"):
        """
        A worker task that continuously pulls (url, depth) from the queue,
        crawls it if eligible, and enqueues new links until max_pages is reached.
        """
        while True:
            try:
                url, depth = await queue.get()
                
                # If we've already crawled enough pages, just mark done and continue
                if self.pages_crawled >= self.max_pages:
                    queue.task_done()
                    continue
                
                if not self.should_crawl(url):
                    queue.task_done()
                    continue

                # Mark as visited
                self.visited_urls.add(url)

                # Respect delay between requests
                await asyncio.sleep(self.delay)

                # Fetch URL
                logger.debug(f"Fetching {url} at depth {depth}")
                html, error = await self.fetch_url(url)
                if error:
                    logger.warning(f"Failed to fetch {url}: {error}")
                    self.failed_urls.append((url, error))
                    queue.task_done()
                    continue

                # Process page
                page_data = self.process_html(html, url)
                self.add_page_to_xml(page_data)

                # Update stats
                self.pages_crawled += 1
                if self.pbar:
                    self.pbar.update(1)
                
                # Extract and enqueue links if not at max depth
                if depth < self.max_depth:
                    soup = BeautifulSoup(html, 'lxml')
                    links = self.extract_links(soup, url)
                    for link in links:
                        # Only enqueue if we haven't exceeded max_pages
                        if self.pages_crawled < self.max_pages:
                            await queue.put((link, depth + 1))

                queue.task_done()
            
            except asyncio.CancelledError:
                # Worker is being canceled (e.g., on shutdown)
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                queue.task_done()

    async def crawl(self) -> None:
        """Main crawl function using concurrency."""
        try:
            # Initialize session
            await self.init_session()
            
            # Create XML document
            self.create_xml_document()

            # Prepare the queue with our initial URL
            queue: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
            await queue.put((self.start_url, 0))

            # Set up progress bar (we'll approximate total as max_pages to start)
            self.pbar = tqdm.tqdm(total=self.max_pages, desc="Crawling", unit="page")

            # Create and start worker tasks
            tasks = []
            for _ in range(self.concurrency):
                t = asyncio.create_task(self.worker(queue))
                tasks.append(t)

            # Wait until the queue is fully processed or we've reached max_pages
            await queue.join()

            # Cancel any remaining worker tasks
            for t in tasks:
                t.cancel()

            # Close progress bar
            self.pbar.close()

            # Save XML to file
            self.save_xml_to_file()
            
        except KeyboardInterrupt:
            logger.info("Crawl interrupted by user")
            print(f"\n{Fore.YELLOW}Crawl interrupted. Saving progress...{Style.RESET_ALL}")
            
            # Save partial results
            if self.xml_root and self.pages_crawled > 0:
                self.save_xml_to_file()
                
        except Exception as e:
            logger.error(f"Crawl error: {e}")
            print(f"{Fore.RED}Error during crawl: {e}{Style.RESET_ALL}")
            
        finally:
            # Close progress bar if still open
            if self.pbar:
                self.pbar.close()
                
            # Close session
            await self.close_session()
            
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DocCrawler: A CLI utility to crawl website documentation and archive it in XML format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl a website documentation")
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
    crawl_parser.add_argument("--extract-headings", action="store_true", default=True, help="Extract and hierarchically organize headings")
    crawl_parser.add_argument("--follow-links", action="store_true", help="Follow links to external domains")
    crawl_parser.add_argument("--clean-html", action="store_true", default=True, help="Enhance cleaning of HTML content")
    crawl_parser.add_argument("--strip-js", action="store_true", default=True, help="Remove JavaScript content")
    crawl_parser.add_argument("--strip-css", action="store_true", default=True, help="Remove CSS content")
    crawl_parser.add_argument("--strip-comments", action="store_true", default=True, help="Remove HTML comments")
    crawl_parser.add_argument("--robots-txt", action="store_true", default=False, help="Respect robots.txt rules")
    
    # NEW: concurrency argument
    crawl_parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent requests")

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    return args

async def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    if args.command == "version":
        print("DocCrawler version 1.0")
        return
        
    if args.command == "crawl":
        # Print banner
        banner = f"""
{Fore.CYAN}
 ____              ____                    _           
|  _ \\  ___   ___ / ___|_ __ __ ___      _| | ___ _ __ 
| | | |/ _ \\ / __| |   | '__/ _` \\ \\ /\\ / / |/ _ \\ '__|
| |_| | (_) | (__| |___| | | (_| |\\ V  V /| |  __/ |   
|____/ \\___/ \\___|\\____|_|  \\__,_| \\_/\\_/ |_|\\___|_|   
{Style.RESET_ALL}
A CLI utility to crawl website documentation and archive it in XML format for LLM ingestion.
Starting crawl from: {Fore.GREEN}{args.start_url}{Style.RESET_ALL}
        """
        print(banner)
        
        # Create and run crawler
        crawler = DocCrawler(args)
        await crawler.crawl()

if __name__ == "__main__":
    # Run with asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
