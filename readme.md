# docs2xml

**docs2xml** is a command-line utility that crawls website documentation (or any set of web pages) and archives it in **XML** format. This is useful for ingestion into LLM (Large Language Model) pipelines or other text analysis workflows.

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

---

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

---

## Usage

```bash
python docs2xml.py crawl <start_url> [options]
```
*(Note: you can also omit the `crawl` keyword—it's inserted by default.)*

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
| `--include-images`          | `False`            | Extract `<img>` tags into the XML output with `alt` text.                                         |
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

---

## Examples

Below are some real-world use case examples using documentation sites.

### 1. **Basic Crawl**

```bash
python docs2xml.py https://docs.mistral.ai/getting-started
```
- Crawls `https://docs.mistral.ai/getting-started` with default settings (max-depth=5, max-pages=1000, concurrency=5).
- Extracts content into `docs_archive.xml`.

---

### 2. **Restrict by Path**

```bash
python docs2xml.py https://docs.mistral.ai/getting-started --restrict-path
```
- Strictly crawls pages **only** whose path begins with `/getting-started`.  
- Any links pointing above `/getting-started` or to another domain are ignored.

---

### 3. **Include Pattern**: Narrow to a Sub-Section

```bash
python docs2xml.py https://docs.anthropic.com/en/prompt-library/ --include-pattern="prompts" --max-depth=3
```
- This command starts at `https://docs.anthropic.com/en/prompt-library/`.
- Only crawls URLs (up to 3 levels deep) **if** they contain the substring "`prompts`" in them.
- Useful if you only want, say, specialized pages that mention “prompts” in their URL.

---

### 4. **Exclude Pattern**: Skip Certain Sections

```bash
python docs2xml.py https://ai.google.dev/gemini-api/docs/ --exclude-pattern="changelog|release-notes"
```
- Begins crawling from `https://ai.google.dev/gemini-api/docs/`.
- Skips any pages that match `changelog` or `release-notes` in the URL.
- Allows you to **avoid** less relevant sections or noisy pages.

---

### 5. **Follow External Links** + Depth Limit

```bash
python docs2xml.py https://platform.openai.com/docs/api-reference/chat/ --follow-links --max-depth=2
```
- Follows external links from `platform.openai.com` to other domains if encountered.
- Limits recursion depth to 2, preventing the crawl from going too deep across external sites.
- Potentially collects references to supporting pages outside the main domain.

---

### 6. **Concurrency & Delay Tweaks**

```bash
python docs2xml.py https://platform.openai.com/docs/api-reference/chat/ --concurrency=10 --delay=0.1
```
- Uses 10 asynchronous worker tasks to speed up crawling.
- Reduces the delay between each request to 0.1 seconds (be mindful of server load).

---

### 7. **Regex Examples for Include Pattern**

Here are some more specific regex patterns you might use in `--include-pattern`:
**Remember when using Regex make sure to escape the . so that it’s interpreted literally (e.g. software\.ai instead of software.ai).**

1. **Match a Path Prefix**  
   ```bash
   --include-pattern="^https://docs\.anthropic\.com/en/prompt-library/guides"
   ```
   - Only crawl pages whose URL starts with `https://docs.anthropic.com/en/prompt-library/guides...`.

2. **Match Multiple Keywords**  
   ```bash
   --include-pattern="(token|embedding|prompt)"
   ```
   - Includes any URLs containing “token,” “embedding,” or “prompt” in their text.

3. **Match a File Extension**  
   ```bash
   --include-pattern="\.html?$"
   ```
   - Only process URLs ending with `.htm` or `.html`.

4. **Combine with `--restrict-path`**  
   ```bash
   python docs2xml.py https://ai.google.dev/gemini-api/docs/ --restrict-path --include-pattern="advanced"
   ```
   - Only crawls pages under the initial path (`/gemini-api/docs/`) that also contain “advanced” in the URL.

---

## How It Works

1. **Queue-Based Crawler**  
   - The crawler uses an `asyncio.Queue` to manage discovered URLs.
   - A fixed number of workers (`--concurrency`) fetch, parse, and enqueue new links until the queue is empty or `--max-pages` is reached.

2. **HTML Processing**  
   - By default, JavaScript, CSS, and comments are removed.  
   - [readability-lxml](https://github.com/buriy/python-readability) extracts main content for a cleaner result.  
   - Headings, paragraphs, lists, tables, and code blocks are identified and placed into structured blocks in the final XML.

3. **Output in XML**  
   The output (by default `docs_archive.xml`) contains:
   - **`<page>`** elements with:
     - `<title>`  
     - `<meta>` data  
     - `<content>` blocks (headings, paragraphs, lists, code, images, etc.)  
   - A `<metadata>` section with timestamps, total pages crawled, and other crawler details.

4. **Failure Tracking**  
   Any URL that cannot be fetched is recorded with an error message (`--verbose` will show more details).

5. **Clipboard & Token Count**  
   - If `pyperclip` is installed, final XML is automatically copied to your clipboard.  
   - If `tiktoken` is installed, a token count for the XML is printed.

---

## Troubleshooting

- **Missing Dependencies**: Confirm you have installed the required libraries:
  ```bash
  pip install aiohttp beautifulsoup4 lxml readability-lxml langdetect tqdm colorama robots pyperclip tiktoken
  ```
- **Timeout Errors**: Increase `--timeout` or reduce `--concurrency`.
- **Permission Denied** (on Linux/macOS):  
  ```bash
  chmod +x docs2xml.py
  ./docs2xml.py <your_url>
  ```
  Or run via `python docs2xml.py`.
- **Encoding/Locale Issues**: If you see strange characters, consider adjusting your system’s locale or forcing a specific encoding in `aiohttp`.

