import os
from dotenv import load_dotenv

# Load .env ONCE
load_dotenv()

# config values for the rest of the app
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_API_BASE = "https://api.github.com/repos/"
GITHUB_API_RAW = "https://raw.githubusercontent.com/"

# ── File-size limits ────────────────────────────────────────────────────────
MAX_SIZE_KB = int(os.getenv("MAX_SIZE_KB", "200"))   # skip files larger than this

# ── File-selection limits ───────────────────────────────────────────────────
MAX_SELECTED_FILES = int(os.getenv("MAX_SELECTED_FILES", "20"))  # cap per query

# ── Concurrency ─────────────────────────────────────────────────────────────
MAX_CONCURRENT_FETCHES = int(os.getenv("MAX_CONCURRENT_FETCHES", "8"))

# ── Prompt / LLM ────────────────────────────────────────────────────────────
MAX_CHUNKS = 8          # fallback hard limit for merged context
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

# ── Cache ───────────────────────────────────────────────────────────────────
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "8"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "lexical")   # "lexical" | "semantic"

# ── Patterns to always skip ──────────────────────────────────────────────────
# Minified files, source-maps, and other non-semantic assets
SKIP_PATTERNS = [
    ".min.js", ".min.css", ".css.map", ".js.map",
    "-min.js", "-min.css",
    ".bundle.js", ".chunk.js",
]

# Extensions excluded from the tree entirely
EXCLUDE_EXT = [
    ".gif", ".jpg", ".jpeg", ".png", ".mp4", ".svg", ".ico",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",   # fonts
    ".pdf", ".zip", ".tar", ".gz",
    ".gitignore", ".git", ".vscode", ".docker",
    ".docstr", ".docstr.yaml", ".github",
]

# Extensions worth parsing deeply
IMPORTANT_EXT = [".py", ".ipynb", ".md", ".json", ".yaml", ".toml", ".txt"]

# File-name fragments that are always high-priority
IMPORTANT_NAMES = ["readme", "setup", "main", "__init__", "app", "model", "config", "run"]


def get_llm(provider=None, model=None):
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    if provider == "ollama":
        from src.llm.ollama import OllamaChat
        return OllamaChat(
            model=model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            timeout=float(os.getenv("OLLAMA_TIMEOUT", "120")),
        )
    if provider != "gemini":
        raise ValueError("LLM provider must be ollama or gemini")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError("Install requirements-gemini.txt to use Gemini.") from exc

    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY before running repository analysis."
        )

    return ChatGoogleGenerativeAI(
        model=model or GOOGLE_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
        timeout=120,
        max_retries=2,
    )
