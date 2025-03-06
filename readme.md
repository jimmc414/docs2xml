```markdown
# docs2xml

**docs2xml** is a command-line utility that crawls website documentation (or any set of web pages) and archives it in **XML** format. This is particularly useful for ingestion into LLM (Large Language Model) pipelines or other text analysis workflows.

## Key Highlights

- **Recursive Crawling**: Traverses internal (and optionally external) links up to a specified depth.
- **Restrict by Path**: An optional `--restrict-path` switch to limit crawling to URLs whose path starts with the path in your `start_url`.
- **Output in XML**: Captures textual data, headings, code blocks, images, tables, and metadata in a structured XML format.
- **Flexible Extraction**:
  - **Include/Exclude** URLs by regex (`--include-pattern`, `--exclude-pattern`).
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
3. (Optional) **Make Script Executable** on Linux/macOS:
   ```bash
   chmod +x docs2xml.py
   ```

## Usage

```bash
python docs2xml.py crawl <start_url> [options]

# or simply omit 'crawl' since it's inserted by default:
python docs2xml.py <start_url> [options]
```

### Options

| Option                      | Default            | Description                                                                                       |
|-----------------------------|--------------------|---------------------------------------------------------------------------------------------------|
| `--output-file=<file>`      | `docs_archive.xml` | Output XML file.                                                                                  |
| `--max-depth=<depth>`       | `5`                | Maximum link depth to follow.                                                                     |
| `--max-pages=<pages>`       | `1000`             | Maximum number of pages to crawl.                                                                 |
| `--user-agent=<agent>`      | `DocCrawler/1.0`   | The User-Agent string used in HTTP requests.                                                      |
| `--delay=<seconds>`         | `0.2`              | Delay between requests (in seconds) to avoid overwhelming servers.                                |
| `--include-pattern=<regex>` | `None`             | Regex pattern to **include**. If provided, only URLs matching this pattern will be crawled.       |
| `--exclude-pattern=<regex>` | `None`             | Regex pattern to **exclude**. If provided, any URL matching this pattern will be skipped.         |
| `--timeout=<seconds>`       | `30`               | Request timeout (in seconds).                                                                     |
| `--verbose`                 | `False`            | Enable verbose logging for debugging.                                                             |
| `--include-images`          | `False`            | Extract image tags (`<img>`) into the XML output with `alt` text.                                 |
| `--include-code`            | `True`             | Extract code blocks with basic language detection.                                                |
| `--extract-headings`        | `True`             | Organize content under headings hierarchically.                                                  |
| `--follow-links`            | `False`            | Follow links to **external** domains. By default, only the starting domain is crawled.            |
| `--clean-html`              | `True`             | Use readability-lxml to focus on main content; remove extraneous HTML.                            |
| `--strip-js`                | `True`             | Remove `<script>` tags entirely.                                                                  |
| `--strip-css`               | `True`             | Remove `<style>` tags entirely.                                                                   |
| `--strip-comments`          | `True`             | Remove HTML comments (`<!-- ... -->`).                                                            |
| `--robots-txt`              | `False`            | Respect `robots.txt` rules when crawling.                                                         |
| `--concurrency=<N>`         | `5`                | Number of concurrent requests (async workers).                                                    |
| `--restrict-path`           | `False`            | Restrict crawling to paths that **start** with your `start_url` path.                             |

### Examples

---

#### 1. Basic Crawl
```bash
python docs2xml.py https://docs.anthropic.com
```
Crawls `https://docs.anthropic.com` with default settings: depth = 5, pages = 1000, concurrency = 5.

---

#### 2. Enabling the `--restrict-path` Switch
```bash
python docs2xml.py crawl https://docs.software.ai/getting-started \
    --restrict-path
```
Only crawls pages **whose path begins** with `/getting-started`. E.g., `https://docs.software.ai/getting-started/foo`, etc.

---

#### 3. Regex Filtering with `--include-pattern`
**Regex** can be used to filter in only certain URLs:
```bash
python docs2xml.py crawl https://example.com \
  --include-pattern="docs"
```
Only crawls URLs that contain `docs`.

##### More Extensive Regex Examples

1. **Match a Specific Subpath**  
   ```bash
   --include-pattern="^https://example\.com/docs/"
   ```
   - `^` anchors the start of the string.
   - Matches any URL that begins with `https://example.com/docs/`.

2. **Match Multiple Paths**  
   ```bash
   --include-pattern="(tutorial|guide)"
   ```
   - Matches URLs containing either `tutorial` or `guide`.

3. **Match Specific File Extensions**  
   ```bash
   --include-pattern="\.html?$"
   ```
   - Matches URLs ending with `.htm` or `.html`.

4. **Match Query Parameters**  
   ```bash
   --include-pattern="\?section="
   ```
   - Only includes URLs that have `?section=` in the query string.

5. **Combine with `--restrict-path`**  
   ```bash
   python docs2xml.py crawl https://docs.example.com/start \
       --restrict-path \
       --include-pattern="^https://docs\.example\.com/start/tutorials"
   ```
   - Ensures we only crawl the path `/start/...`
   - Further narrows it to URLs containing `/start/tutorials...`

---

#### 4. Excluding Certain URLs
```bash
python docs2xml.py https://example.com \
  --exclude-pattern="blog|login"
```
Skips URLs that contain the words `blog` or `login`.

---

#### 5. Following External Links
```bash
python docs2xml.py crawl https://docs.anthropic.com \
  --follow-links \
  --max-depth=2
```
Includes external domains and limits recursion depth to 2.

---

#### 6. Concurrency & Delay Tweaks
```bash
python docs2xml.py https://docs.anthropic.com \
  --concurrency=10 \
  --delay=0.1
```
Uses 10 worker tasks with a 0.1s delay.

---

### How It Works

1. **Queue-Based Crawler**  
   An `asyncio.Queue` manages discovered URLs. A fixed number of workers (`--concurrency`) fetch, parse, and enqueue new links.

2. **HTML Processing**  
   - By default, JavaScript, CSS, and comments are removed.  
   - [readability-lxml](https://github.com/buriy/python-readability) extracts main content for a cleaner result.  
   - Headings, paragraphs, lists, tables, and code blocks are identified and placed into structured blocks in the final XML.

3. **Output in XML**  
   The default `docs_archive.xml` contains:
   - **`<page>`** elements with:
     - `<title>`  
     - `<meta>` data  
     - `<content>` blocks (headings, paragraphs, lists, code, images, etc.)  
   - A `<metadata>` section with timestamps, total pages crawled, and other crawler details.

4. **Failure Tracking**  
   Any URL that cannot be fetched is recorded with an error message. If `--verbose` is on, more detail is displayed.

5. **Clipboard & Token Count**  
   - If `pyperclip` is installed, the final XML is copied to your clipboard automatically.  
   - If `tiktoken` is installed, a token count is printed.

## Troubleshooting

- **Missing Dependencies**: Ensure all required libraries are installed:
  ```bash
  pip install aiohttp beautifulsoup4 lxml readability-lxml langdetect tqdm colorama robots pyperclip tiktoken
  ```
- **Timeout Errors**: Increase `--timeout` or reduce `--concurrency`.
- **Permission Denied** (Linux/macOS): Run `chmod +x docs2xml.py` or call `python docs2xml.py` directly.
- **Encoding/Locale Issues**: If you encounter strange characters, consider adjusting your system’s locale or forcibly specifying response encoding in `aiohttp`.
