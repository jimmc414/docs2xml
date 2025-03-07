# docs2xml

A **command-line utility** that crawls website documentation (or any set of web pages) and archives it in **XML** format for subsequent use with LLMs or text analysis workflows.  
It also supports **chunking** and **embedding** the resulting XML into a local [ChromaDB](https://docs.trychroma.com/) vector database.

---

## Key Features

- **Recursive Crawling**  
  - Traverses internal (and optionally external) links up to a specified depth.
  - *Optional:* Restrict paths via `--restrict-path`.
- **Output in XML**  
  - Captures textual data, headings, code blocks, images, tables, etc.
- **Flexible Extraction**  
  - **Include/Exclude** URLs by regex (`--include-pattern`, `--exclude-pattern`).
  - **Strip** JavaScript, CSS, or HTML comments.
  - **Extract** headings hierarchically or just as paragraphs/lists.
  - **Include** code blocks (with basic language detection) and/or images.
- **Robots.txt Respect** (optional).
- **Concurrent Async Requests** (`--concurrency`).
- **Clipboard Copy & Token Counting**  
  - If `pyperclip` is installed, the final XML is automatically copied to your clipboard.  
  - If `tiktoken` is installed, a token count of the XML is shown at the end.

**New in v2.0**:

- **Chunk & Embed** your crawled XML (or an existing XML) into ChromaDB:
  - Chunk by **token** or **character** length (`--chunk-by-chars`).
  - Various embedding functions: `default`, `openai`, `cohere`, `huggingface`.
  - Store embeddings in a local ChromaDB (`--chroma-collection`, `--chroma-persist-dir`).
- **`--chunk-and-embed`** for a **one-step** workflow:
  - Crawl -> XML -> Chunk -> Embed in ChromaDB
- **Or** use `embed` subcommand on an existing XML file.

---

## Installation

1. **Clone or Download** the `docs2xml.py` file.
2. **Install Base Dependencies**:
   ```bash
   pip install aiohttp beautifulsoup4 lxml readability-lxml langdetect tqdm colorama robots pyperclip tiktoken
   ```
   > *Note:* `pyperclip` and `tiktoken` are optional but recommended for copy-to-clipboard and token counting, respectively.
3. **Install ChromaDB** (required for embedding):
   ```bash
   pip install chromadb numpy
   ```
4. **Install Additional Embedding Providers** if needed:
   - **OpenAI**:
     ```bash
     pip install openai
     ```
   - **Cohere**:
     ```bash
     pip install cohere
     ```
   - **HuggingFace**:
     ```bash
     pip install sentence-transformers
     ```

*(You only need to install the packages for whichever embedding provider you choose.)*

---

## Usage

The script supports **two primary commands**:  
1. **`crawl`**: Crawl a website and output XML (optionally chunk+embed).  
2. **`embed`**: Chunk+embed an existing XML file.

You can always run:
```bash
python docs2xml.py --help
```
or
```bash
python docs2xml.py <command> --help
```
to see all options.

---

### 1. Crawling

```bash
python docs2xml.py crawl <start_url> [options]
```

**Basic Example**:
```bash
python docs2xml.py crawl https://docs.mistral.ai/getting-started
```
- Crawls the given URL (and internal links) up to default depth and pages.
- Outputs to `docs_archive.xml` by default.

**Important Options**:

| Option                          | Default            | Description                                                                                       |
|---------------------------------|--------------------|---------------------------------------------------------------------------------------------------|
| `--output-file=<file>`          | `docs_archive.xml` | Output XML file                                                                                  |
| `--max-depth=<depth>`           | `5`                | Maximum link depth to follow                                                                     |
| `--max-pages=<pages>`           | `1000`             | Maximum pages to crawl                                                                           |
| `--user-agent=<agent>`          | `DocCrawler/1.0`   | HTTP User-Agent string                                                                           |
| `--delay=<seconds>`             | `0.2`              | Delay between requests                                                                           |
| `--include-pattern=<regex>`     | `None`             | Include only URLs matching this regex                                                            |
| `--exclude-pattern=<regex>`     | `None`             | Exclude URLs matching this regex                                                                 |
| `--timeout=<seconds>`           | `30`               | Request timeout                                                                                  |
| `--verbose`                     | `False`            | Enable verbose logging                                                                           |
| `--include-images`              | `False`            | Extract `<img>` tags into XML (with `alt` text)                                                 |
| `--include-code`                | `True`             | Extract code blocks with naive language detection                                                |
| `--extract-headings`            | `True`             | Organize content under headings                                                                  |
| `--follow-links`                | `False`            | Follow external links                                                                            |
| `--clean-html`                  | `True`             | Use readability-lxml to filter out extraneous HTML                                              |
| `--strip-js`                    | `True`             | Remove `<script>` tags                                                                           |
| `--strip-css`                   | `True`             | Remove `<style>` tags                                                                            |
| `--strip-comments`              | `True`             | Remove HTML comments                                                                             |
| `--robots-txt`                  | `False`            | Respect robots.txt rules                                                                         |
| `--concurrency=<N>`             | `5`                | Number of concurrent async requests                                                              |
| `--restrict-path`               | `False`            | Restrict URLs to the initial path of your start_url                                              |

**Extra**:
- `--restrict-path` ensures only pages whose path matches (or is under) the start URL path are crawled.  

---

### 2. Chunk and Embed with ChromaDB

**docs2xml** v2.0 adds optional **chunking** and **embedding** to store content in a local ChromaDB.  

#### Approach A: **Crawl & Embed in One Step**
```bash
python docs2xml.py crawl https://docs.example.com \
  --chunk-and-embed \
  --chunk-size=512 --chunk-overlap=128 \
  --embedding-function=openai \
  --openai-api-key=sk-...
```
**What Happens**:
1. Crawls site -> saves to XML (`docs_archive.xml` by default).  
2. Splits text into ~512-token chunks (128-token overlap).  
3. Embeds each chunk using **OpenAI** embeddings.  
4. Stores chunks in a ChromaDB collection (`documentation` by default in `./chroma_db`).  

#### Approach B: **Embed an Existing XML**
If you already have an XML file from a previous crawl (or from some other source):
```bash
python docs2xml.py embed docs_archive.xml \
  --chunk-size=1024 \
  --chunk-overlap=128 \
  --embedding-function=huggingface \
  --huggingface-model-name="sentence-transformers/all-MiniLM-L6-v2" \
  --chroma-collection="my_docs" \
  --output-chunks=chunks.json
```
**What Happens**:
1. **Reads** `docs_archive.xml`, merges page content into large strings.  
2. Splits them into chunks of 1024 tokens, with 128-token overlap.  
3. Embeds chunks using HuggingFace (local model).  
4. Stores them in the `my_docs` ChromaDB collection, located at `./chroma_db`.  
5. Also saves a JSON file (`chunks.json`) with the chunk text & metadata.

---

### Chunking Options

| Option                      | Default      | Description                                                                                                         |
|-----------------------------|--------------|---------------------------------------------------------------------------------------------------------------------|
| `--chunk-size=<int>`       | `1024`       | Size of each chunk in tokens (or chars if `--chunk-by-chars` is set)                                               |
| `--chunk-overlap=<int>`    | `128`        | Overlap length between consecutive chunks (tokens or chars)                                                        |
| `--chunk-by-chars`         | `False`      | If set, chunk by **characters** instead of tokens                                                                  |
| `--ignore-content-boundaries` | `False`   | Split purely by chunk size, ignoring paragraph/heading boundaries                                                  |
| `--output-chunks=<file>`   | (Not saved)  | Write chunk data to a JSON file for inspection                                                                     |

> **Token-based chunking** uses [tiktoken](https://github.com/openai/tiktoken) by default. If `tiktoken` isn’t installed or `--chunk-by-chars` is given, it will fall back to character-based chunking.

---

### Embedding Function Options

Select among:
1. **default**  
   - A built-in placeholder embedding from Chroma (no API key, lower quality).
2. **openai**  
   - Requires `--openai-api-key`.  
   - Installs `openai` library.  
   - Uses `text-embedding-ada-002`.
3. **cohere**  
   - Requires `--cohere-api-key`.  
   - Installs `cohere` library.
4. **huggingface**  
   - Requires `--huggingface-model-name=<model>`.  
   - Installs `sentence-transformers`.  
   - Runs locally (e.g. `"sentence-transformers/all-MiniLM-L6-v2"`).

**ChromaDB**:
- `--chroma-collection=<name>` (default: `"documentation"`)  
- `--chroma-persist-dir=<path>` (default: `./chroma_db`)

---

## Examples

1. **Basic Crawl Only**  
   ```bash
   python docs2xml.py crawl https://docs.mistral.ai/getting-started
   ```
   - Outputs to `docs_archive.xml`.

2. **Restrict by Path**  
   ```bash
   python docs2xml.py crawl https://docs.mistral.ai/getting-started --restrict-path
   ```
   - Only crawls URLs whose path begins with `/getting-started`.

3. **Include/Exclude Patterns**  
   ```bash
   python docs2xml.py crawl https://ai.google.dev/gemini-api/docs/ \
     --include-pattern="advanced" --exclude-pattern="changelog|release-notes"
   ```
   - Only crawls pages with "advanced" in the URL.
   - Skips pages that mention "changelog" or "release-notes".

4. **Follow External Links, Limit Depth**  
   ```bash
   python docs2xml.py crawl https://platform.openai.com/docs/api-reference/chat/ --follow-links --max-depth=2
   ```
   - Goes up to 2 levels deep, even if links lead off the main domain.

5. **Chunk & Embed Immediately**  
   ```bash
   python docs2xml.py crawl https://docs.example.com \
     --chunk-and-embed \
     --chunk-size=512 --chunk-overlap=128 \
     --embedding-function=openai --openai-api-key=sk-...
   ```
   - Creates `docs_archive.xml`.
   - Chunks ~512 tokens, 128 overlap.
   - Embeds with OpenAI, stores in `documentation` collection in `./chroma_db`.

6. **Embed an Existing XML**  
   ```bash
   python docs2xml.py embed docs_archive.xml \
     --chunk-size=1024 --chunk-overlap=256 \
     --embedding-function=huggingface \
     --huggingface-model-name="sentence-transformers/all-MiniLM-L6-v2" \
     --chroma-collection="product_docs" \
     --output-chunks=chunks.json
   ```
   - Processes `docs_archive.xml`.
   - Splits into ~1024-token chunks, 256 overlap.
   - Embeds locally with HuggingFace model.
   - Saves chunks info to `chunks.json`.
   - Stores in the `product_docs` collection.

---

## Retrieving Documents from Chroma

After embedding, you can use [ChromaDB’s Python API](https://docs.trychroma.com/embeddings) to query relevant chunks. For example:

```python
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Same embedding function used during chunking
embedding_function = embedding_functions.DefaultEmbeddingFunction()

client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_collection("documentation", embedding_function=embedding_function)

query_text = "How do I install the software?"
results = collection.query(
    query_texts=[query_text],
    n_results=3
)

for i, document in enumerate(results["documents"][0]):
    print(f"Result {i+1}: {document}")
```

---

## Troubleshooting

- **Missing Dependencies**  
  Make sure you installed the required libraries, including `chromadb` and any optional embedding packages:
  ```bash
  pip install chromadb openai cohere sentence-transformers
  ```
  *(Depending on what you actually use.)*
- **Timeout or Rate Limits**  
  Adjust `--timeout` or add `--delay`. For big sites, consider raising `--max-pages`.
- **Encoding Issues**  
  Try specifying a UTF-8 locale or adjusting the default encoding if you see weird characters.
- **Clipboard & Token Count**  
  - Requires `pyperclip` for copying XML to your clipboard.
  - Requires `tiktoken` for token counting.
