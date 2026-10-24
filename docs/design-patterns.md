# Design Patterns & System Architecture

This document explains the structural design patterns and their role in the system.

---

## 1. Factory Pattern: Category → Loader

**Location:** `src/ingestion/loader/loader.py`

```python
LOADERS = {
    "cars": CarsLoader,
    "motorcycles": MotorcyclesLoader,
    "scooters": ScootersLoader,
    "spare_parts": SparePartsLoader
}
```

**Used in:** `src/ingestion/pipeline.py::run_ingestion`

```python
for category in LOADERS:
    loader_class = LOADERS[category]
    raw_docs = await loader_class().download_and_extract()
```

### How it works

1. **Category name** (string) → **Loader class** (callable)
2. Instantiate the loader, call `download_and_extract()`
3. No conditional logic needed; the dict handles routing

### Why not a switch?

```python
# ❌ Without factory (scales poorly)
if category == "cars":
    loader = CarsLoader()
elif category == "motorcycles":
    loader = MotorcyclesLoader()
# ... add every new loader here

# ✅ With factory (extensible)
loader = LOADERS[category]()
# new loader? Add one line to LOADERS dict
```

### Trade-off

**Pro:** Adding a 5th loader (EVs, trucks, etc.) requires one line: `"evs": EVsLoader`.  
**Con:** All loaders must share the same interface (`download_and_extract()`).

---

## 2. Template Method Pattern: BaseLoader → Subclasses

**Location:** `src/ingestion/loader/base_loader.py` (skeleton) + subclass overrides

### The skeleton (BaseLoader)

```python
class BaseLoader:
    async def download_and_extract(self):
        """Crawl all SOURCES, discover product links, extract each, return docs."""
        for source in self.SOURCES:
            try:
                async with await async_playwright() as playwright:
                    browser = await playwright.chromium.launch()
                    context = await browser.new_context()
                    page = await context.new_page()
                    
                    # Step 1: Discover product links
                    links = await self.discover_product_links(page, source)  # HOOK
                    
                    # Step 2: Extract each product
                    for link in links:
                        raw_doc = await self.extract_product_page(page, link)  # HOOK
                        if raw_doc:
                            yield raw_doc
```

### The hooks (filled by subclasses)

```python
class CarsLoader(BaseLoader):
    """Tata Cars-specific discovery."""
    async def discover_product_links(self, page, source):
        await page.goto(source["url"])
        # Tata.cars structure: /model/energy.html
        models = ["nexon", "altroz", "harrier"]
        energies = ["ice", "ev", "cng"]
        return [f"/{m}/{e}.html" for m in models for e in energies]

    async def extract_product_page(self, page, product_url):
        await page.goto(product_url)
        # Extract via Tata.cars CSS selectors
        return {
            "title": await page.locator("h1").inner_text(),
            "price": await page.locator("[class*='price']").inner_text(),
            # ...
        }

class GenericVehicleLoader(BaseLoader):
    """Hero shop: generic discovery (follow links)."""
    async def discover_product_links(self, page, source):
        await page.goto(source["url"])
        # Click and find all product links
        return await page.evaluate(
            "document.querySelectorAll('a.product-link') |> [...].map(a => a.href)"
        )

    async def extract_product_page(self, page, product_url):
        # Generic CSS selectors (work across Hero motorcycles and scooters)
        return {
            "title": await page.locator("[class*='product-name']").inner_text(),
            # ...
        }
```

### Why this pattern?

| Concern | Handled by |
|---------|-----------|
| Browser lifecycle (launch, close) | BaseLoader (all loaders benefit) |
| Error handling (one bad product doesn't fail the run) | BaseLoader savepoint |
| Image download, validation, vision captioning | BaseLoader `process_image` |
| Site-specific discovery (where are products?) | Subclass `discover_product_links` |
| Site-specific extraction (how to read a product page?) | Subclass `extract_product_page` |

**Without this pattern:** Each loader copies the whole 200-line crawl loop, then changes discovery + extraction. Fix a browser bug? Update 4 copies. **With this pattern:** Fix once in BaseLoader, all loaders benefit.

### Trade-off

**Pro:** Code reuse, single source of truth for crawl logic.  
**Con:** Must understand BaseLoader's async/await flow before adding a loader. Takes reading the base class.

---

## 3. Lazy Singleton Pattern: Embedder

**Location:** `src/ingestion/embedding/embedder.py`

```python
class Embedder:
    """Thin Ollama HTTP client."""
    def __init__(self):
        self.embed_model = "nomic-embed-text"
        self.vision_model = "llava"
        # Shares one HTTP session across all calls
```

**Used in:** `src/ingestion/pipeline.py`

```python
async def run_ingestion(...):
    embedder = Embedder()  # One instance per run
    # Pass to loaders
    loader = GenericVehicleLoader(embedder=embedder)
```

**Used in loaders:** `src/ingestion/loader/generic_loader.py`

```python
class GenericVehicleLoader(BaseLoader):
    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()  # Lazy: create if not provided
```

### Why "Lazy Singleton"?

- **Lazy:** Not created at module import; only when needed (`embedder = Embedder()` in pipeline)
- **Singleton per run:** One embedder instance per pipeline run, shared across all loaders
- **Not a true singleton:** Different runs have different embedders (no global state)

### Why not a true singleton?

```python
# ❌ True singleton (Python pattern)
class Embedder:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**Problems:**
- Async code: shared state → thread-safety issues
- Testing: can't create isolated instances
- Cleanup: singleton.close() must be called explicitly, or it hangs

### The alternative we use: dependency injection

```python
# ✅ Pass the embedder as an argument
loader = GenericVehicleLoader(embedder=embedder)
# Testing: mock embedder
test_loader = GenericVehicleLoader(embedder=MockEmbedder())
```

---

## 4. Connection Pool Pattern: MetadataStore + asyncpg

**Location:** `src/ingestion/metadata/metadata_store.py`

```python
class MetadataStore:
    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def init_db(self):
        self.pool = await asyncpg.create_pool(
            **DB_PARAMS,
            min_size=5,  # Min connections
            max_size=20  # Max connections
        )

    async def upsert_documents(self, docs, ...):
        async with self.pool.acquire() as conn:  # Borrow a connection
            async with conn.transaction():
                for doc in docs:
                    await conn.execute("INSERT INTO brochure_chunks (...) VALUES (...)")
```

### How it works

1. **create_pool()** pre-allocates 5 connections to the database
2. **acquire()** borrows one from the pool
3. Use it (execute queries)
4. **Release** (automatic via `async with`) returns it to the pool
5. Next query reuses the same connection (no TCP handshake overhead)

### Why asyncpg, not psycopg2?

| Library | Type | Suitable for |
|---------|------|---|
| psycopg2 | Sync, blocking | CLI tools, batch scripts |
| asyncpg | Async, non-blocking | Web servers, event-loop apps |

**In this project:** Retrieval runs per-request (latency-sensitive). Each 35ms Ollama embedding call + 20ms database query adds up. Blocking psycopg2 would block the entire event loop; asyncpg doesn't.

---

## 5. Where These Patterns Are Missing

### Dependency Injection Container

Today, dependencies are threaded manually:

```python
embedder = Embedder()
loader = GenericVehicleLoader(embedder=embedder)
store = MetadataStore()
```

At scale (10 loaders, 5 embedder types), this gets verbose. A simple DI container would centralize wiring:

```python
container = Container()
container.register(Embedder, single=True)  # Shared across loaders
loader = container.resolve(GenericVehicleLoader)
```

**When to add:** When the number of wired dependencies exceeds ~10 or when testing becomes painful.

### Adapter Pattern for Document Extraction

Each loader's `extract_product_page` builds a dict with site-specific keys:

```python
# CarsLoader
{"title": "...", "price": "...", "features": [...]}

# SparePartsLoader
{"part_name": "...", "price": "...", "vehicle_types": [...]}
```

An Adapter could normalize HTML → one dict shape:

```python
class ProductExtractor(ABC):
    @abstractmethod
    async def extract(self, page: Page) -> ProductDict:
        pass

class TataCarsExtractor(ProductExtractor):
    async def extract(self, page):
        return {
            "title": await page.locator("h1").inner_text(),
            ...
        }
```

**When to add:** When loaders copy selector logic or when the normalized shape changes often.

---

## Summary

| Pattern | Purpose | Cost |
|---------|---------|------|
| Factory | Route category → loader class | One dict |
| Template Method | Share crawl loop, customize discovery/extraction | Must learn BaseLoader |
| Lazy Singleton | One Ollama connection per run | Requires manual threading in tests |
| Connection Pool | Reuse DB connections, avoid TCP overhead | asyncpg syntax (learning curve) |

**Principle:** Use patterns to reduce duplication and improve testability, but don't over-abstract. The codebase today has 4 loaders and 1 run-per-day; fancy DI containers would be premature.
