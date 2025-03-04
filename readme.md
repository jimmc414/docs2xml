# docs2xml

**docs2xml** is a command-line utility that crawls website documentation (or any set of web pages) and archives it in **XML** format. This is particularly useful for ingestion into LLM (Large Language Model) pipelines or other text analysis workflows.

## Key Highlights

- **Recursive Crawling**: Traverses internal (and optionally external) links up to a specified depth.
- **Output in XML**: Captures textual data, headings, code blocks, images, tables, and metadata in a structured XML format.
- **Flexible Extraction**:
  - **Include/Exclude** URLs by regex.
  - **Strip** JavaScript, CSS, or HTML comments.
  - **Extract** headings hierarchically.
  - **Include** code blocks with basic language detection.
  - **Include** images with `alt` descriptions.
- **Robots.txt Respect**: Optionally honor robots.txt rules to restrict crawling.
- **Concurrent Requests**: Speed up crawling with configurable concurrency.
- **Cleaned Output**: Leverages [readability-lxml](https://github.com/buriy/python-readability) to extract main content and supports additional cleaning of HTML.

## Installation

1. **Clone or Download** the `docs2xml.py` file.
2. **Install Required Dependencies**:
   ```bash
   pip install aiohttp beautifulsoup4 lxml readability-lxml langdetect tqdm colorama robots pyperclip tiktoken
   ```

### Make the Script Executable (Linux / macOS)

If you are on Linux or macOS, make the script executable:
```bash
chmod +x docs2xml.py
```
Then you can run:
```bash
./docs2xml.py <start_url> [options]
```

Otherwise, on any platform (including Windows), you can simply run:
```bash
python docs2xml.py <start_url> [options]
```

## Usage

**Important:** Adding the `crawl` verb is optional because the script defaults to `crawl` if no subcommand is specified. You can run the crawler in either of the following ways:

```bash
# Default usage (no subcommand, auto-crawl)
python docs2xml.py <start_url> [options]

# or explicitly specify "crawl"
python docs2xml.py crawl <start_url> [options]
```

You may also run:
```bash
python docs2xml.py --help
```
for more details.

### Options

| Option                 | Default           | Description                                                                                       |
|------------------------|-------------------|---------------------------------------------------------------------------------------------------|
| `--output-file=<file>` | `docs_archive.xml`| Output XML file.                                                                                  |
| `--max-depth=<depth>`  | `5`               | Maximum link depth to follow.                                                                     |
| `--max-pages=<pages>`  | `1000`            | Maximum number of pages to crawl.                                                                 |
| `--user-agent=<agent>` | `docs2xml/1.0`    | The User-Agent string used in HTTP requests.                                                      |
| `--delay=<seconds>`    | `0.2`             | Delay between requests (in seconds) to avoid overwhelming servers.                                |
| `--include-pattern=<regex>` | `None`       | Regex pattern to **include**. If provided, only URLs matching this pattern will be crawled.       |
| `--exclude-pattern=<regex>` | `None`       | Regex pattern to **exclude**. If provided, any URL matching this pattern will be skipped.         |
| `--timeout=<seconds>`  | `30`              | Request timeout (in seconds).                                                                     |
| `--verbose`            | `False`           | Enable verbose logging for debugging.                                                             |
| `--include-images`     | `False`           | Extract image tags (`<img>`) into the XML output with `alt` text.                                 |
| `--include-code`       | `True`            | Extract code blocks with basic language detection.                                                |
| `--extract-headings`   | `True`            | Organize content under headings hierarchically.                                                  |
| `--follow-links`       | `False`           | Follow links to **external** domains. By default, only the starting domain is crawled.            |
| `--clean-html`         | `True`            | Use readability-lxml to focus on main content; remove extraneous HTML.                            |
| `--strip-js`           | `True`            | Remove `<script>` tags entirely.                                                                  |
| `--strip-css`          | `True`            | Remove `<style>` tags entirely.                                                                   |
| `--strip-comments`     | `True`            | Remove HTML comments (`<!-- ... -->`).                                                            |
| `--robots-txt`         | `False`           | Respect `robots.txt` rules when crawling.                                                         |
| `--concurrency=<N>`    | `5`               | Number of concurrent requests (async workers).                                                    |

### Examples

1. **Basic Crawl (No Subcommand)**  
   ```bash
   python docs2xml.py https://docs.anthropic.com
   ```
   Crawls `https://docs.anthropic.com` with default settings (depth=5, pages=1000, concurrency=5).

2. **Explicit Crawl Verb**  
   ```bash
   python docs2xml.py crawl https://docs.anthropic.com
   ```

3. **Crawl with Include/Exclude Patterns**  
   ```bash
   python docs2xml.py https://example.com \
       --include-pattern="docs" \
       --exclude-pattern="blog" \
       --max-depth=3
   ```
   Crawls only URLs containing “docs” while excluding those with “blog,” to a depth of 3.

4. **Follow External Links**  
   ```bash
   python docs2xml.py crawl https://docs.anthropic.com \
       --follow-links \
       --max-depth=2
   ```
   Includes links outside `example.com`, up to a depth of 2.

5. **Verbose & Respect robots.txt**  
   ```bash
   python docs2xml.py https://docs.anthropic.com \
       --verbose \
       --robots-txt
   ```
   Prints debug info for each request and respects `robots.txt` rules.

6. **Concurrency & Delay Tweaks**  
   ```bash
   python docs2xml.py https://docs.anthropic.com \
       --concurrency=10 \
       --delay=0.1
   ```
   Uses 10 workers to crawl more quickly, with a 0.1s delay between requests.

## How It Works

1. **Queue-Based Crawler**  
   The crawler uses an `asyncio.Queue` to manage discovered URLs. A set number of worker tasks (`--concurrency`) continuously fetch URLs, parse them, and enqueue newly found links.

2. **HTML Processing**  
   - By default, JavaScript, CSS, and comments are removed.  
   - [readability-lxml](https://github.com/buriy/python-readability) extracts the main content for a cleaner result.  
   - Headings, paragraphs, lists, tables, and code blocks are identified and placed into structured blocks.

3. **Output in XML**  
   The final XML (default: `docs_archive.xml`) contains:
   - **`<page>`** elements with:
     - `<title>`  
     - `<meta>` data  
     - `<content>` blocks (headings, paragraphs, lists, code, images, etc.)  
   - A `<metadata>` section with timestamps, total pages crawled, and other crawler details.

4. **Failure Tracking**  
   Any URL that cannot be fetched or processed is recorded with an error message. If `--verbose` is on, more detail is displayed.

5. **Clipboard & Token Count**  
   - If `pyperclip` is installed, the final XML is also copied to your clipboard.  
   - If `tiktoken` is installed, a token count for the XML is printed (useful for LLM context size planning).

## Troubleshooting

- **Missing Dependencies**: Ensure all required libraries are installed:
  ```bash
  pip install aiohttp beautifulsoup4 lxml readability-lxml langdetect tqdm colorama robots pyperclip tiktoken
  ```
- **Timeout Errors**: Increase `--timeout` or lower `--concurrency`.
- **Permission Denied** (Linux/macOS): Run `chmod +x docs2xml.py` or call `python docs2xml.py` directly.
- **Encoding/Locale Issues**: If you encounter strange characters, consider adjusting your system’s locale or using a different approach for decoding in `aiohttp`.
