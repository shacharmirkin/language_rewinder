import os
import gradio as gr
from google import genai
from google.genai import types
from dotenv import load_dotenv
import re
import sqlite3
import hashlib
import threading
import logging
import json
from datetime import datetime, timezone
import time

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"
INPUT_COST_PER_1M_TOKENS_USD = 0.30
OUTPUT_COST_PER_1M_TOKENS_USD = 2.50

def pick_storage_root():
    if os.environ.get("LTM_STORAGE_DIR"):
        return os.environ["LTM_STORAGE_DIR"]
    if os.environ.get("SPACE_ID"):
        # HF persistent storage (paid) is mounted at /data.
        if os.path.isdir("/data") and os.access("/data", os.W_OK):
            return "/data/lang_rewinder"
        return "/tmp/lang_rewinder"
    return "."

storage_root = pick_storage_root()
cache_db_path = os.environ.get(
    "LLM_CACHE_DB_PATH",
    os.path.join(storage_root, ".cache", "lang_rewinder_cache.sqlite3"),
)
log_path = os.environ.get(
    "LLM_LOG_PATH",
    os.path.join(storage_root, ".logs", "lang_rewinder_requests.log"),
)

cache_dir = os.path.dirname(cache_db_path)
if cache_dir:
    os.makedirs(cache_dir, exist_ok=True)

log_dir = os.path.dirname(log_path)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)

cache_conn = sqlite3.connect(cache_db_path, check_same_thread=False)
cache_lock = threading.Lock()
with cache_lock:
    cache_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            output_text TEXT NOT NULL,
            expires_at INTEGER
        )
        """
    )
    try:
        cache_conn.execute("ALTER TABLE llm_cache ADD COLUMN expires_at INTEGER")
    except sqlite3.OperationalError:
        pass
    cache_conn.commit()

logger = logging.getLogger("lang_rewinder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

SYSTEM_PROMPT = """
You are a Historical Linguist and Translator. Your goal is to rewrite modern text into the vocabulary, 
syntax, and slang of a specific year or decade. 

RULES:
1. ANCHRONISM FILTER: Strictly avoid words, concepts, or technologies that did not exist in the target year.
2. CULTURAL VIBE: Adopt the social tone of the era (e.g., the earnestness of the 40s, the groove of the 70s).
3. EXPLANATION: After the translation, provide a short 'Etymology Note' explaining why you replaced certain modern words.
4. LANGUAGE TRANSFER: If the input is not in English, translate it into the target year's equivalent within that same language. Do not translate between languages (e.g., French stays French).
"""

def log_request(event_data):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event_data,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))

def extract_usage_and_cost(response):
    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    output_tokens = getattr(usage, "candidates_token_count", None) if usage else None

    input_cost_usd = None
    output_cost_usd = None
    total_cost_usd = None
    if input_tokens is not None and output_tokens is not None:
        input_cost_usd = (input_tokens / 1_000_000) * INPUT_COST_PER_1M_TOKENS_USD
        output_cost_usd = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M_TOKENS_USD
        total_cost_usd = input_cost_usd + output_cost_usd

    return input_tokens, output_tokens, input_cost_usd, output_cost_usd, total_cost_usd

def get_cached_output(cache_key):
    now_ts = int(time.time())
    with cache_lock:
        row = cache_conn.execute(
            """
            SELECT output_text
            FROM llm_cache
            WHERE cache_key = ?
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            (cache_key, now_ts),
        ).fetchone()
    return row[0] if row else None

def set_cached_output(cache_key, output_text, ttl_seconds=None):
    expires_at = None
    if ttl_seconds is not None:
        expires_at = int(time.time()) + int(ttl_seconds)
    with cache_lock:
        cache_conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, output_text, expires_at) VALUES (?, ?, ?)",
            (cache_key, output_text, expires_at),
        )
        cache_conn.commit()

def translate_text(user_input, target_year):
    if not api_key:
        output_text = "Error: API key not found. Please set GEMINI_API_KEY."
        log_request(
            {
                "model": MODEL_NAME,
                "target_year": target_year,
                "input_text": user_input,
                "output_text": output_text,
                "error_text": "Missing GEMINI_API_KEY",
                "cache_hit": False,
                "input_tokens": None,
                "output_tokens": None,
                "input_cost_usd": None,
                "output_cost_usd": None,
                "total_cost_usd": None,
            }
        )
        return output_text
    if not user_input.strip():
        return ""

    cache_input = f"{MODEL_NAME}|{target_year}|{SYSTEM_PROMPT}|{user_input}"
    cache_key = hashlib.sha256(cache_input.encode("utf-8")).hexdigest()
    cached_output = get_cached_output(cache_key)
    if cached_output is not None:
        log_request(
            {
                "model": MODEL_NAME,
                "target_year": target_year,
                "input_text": user_input,
                "output_text": cached_output,
                "error_text": None,
                "cache_hit": True,
                "input_tokens": None,
                "output_tokens": None,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
            }
        )
        return cached_output
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            ),
            contents=f"Target Year: {target_year}\nText: {user_input}"
        )
        
        # Intercept the response and squash multiple newlines down to just one
        cleaned_text = re.sub(r'\n{2,}', '\n', response.text.strip())
        input_tokens, output_tokens, input_cost_usd, output_cost_usd, total_cost_usd = extract_usage_and_cost(response)
        set_cached_output(cache_key, cleaned_text)
        log_request(
            {
                "model": MODEL_NAME,
                "target_year": target_year,
                "input_text": user_input,
                "output_text": cleaned_text,
                "error_text": None,
                "cache_hit": False,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost_usd": input_cost_usd,
                "output_cost_usd": output_cost_usd,
                "total_cost_usd": total_cost_usd,
            }
        )
        
        return cleaned_text
        
    except Exception as e:
        error_text = str(e)
        is_unavailable_error = "503" in error_text and "UNAVAILABLE" in error_text
        if is_unavailable_error:
            output_text = (
                "🚦 The model is under high demand right now. Please try again in a minute."
            )
            set_cached_output(cache_key, output_text, ttl_seconds=30)
            log_request(
                {
                    "model": MODEL_NAME,
                    "target_year": target_year,
                    "input_text": user_input,
                    "output_text": output_text,
                    "error_text": error_text,
                    "cache_hit": False,
                    "input_tokens": None,
                    "output_tokens": None,
                    "input_cost_usd": None,
                    "output_cost_usd": None,
                    "total_cost_usd": None,
                }
            )
            return output_text

        is_quota_error = "RESOURCE_EXHAUSTED" in error_text or "429" in error_text
        if is_quota_error:
            is_daily_quota = "PerDay" in error_text or "free_tier_requests" in error_text
            if is_daily_quota:
                output_text = (
                    "🚫 Daily free-tier quota reached for this model. Waiting a few seconds won't help. "
                    "Please try again after the daily reset, switch model, or use a paid quota."
                )
                set_cached_output(cache_key, output_text, ttl_seconds=300)
                log_request(
                    {
                        "model": MODEL_NAME,
                        "target_year": target_year,
                        "input_text": user_input,
                        "output_text": output_text,
                        "error_text": error_text,
                        "cache_hit": False,
                        "input_tokens": None,
                        "output_tokens": None,
                        "input_cost_usd": None,
                        "output_cost_usd": None,
                        "total_cost_usd": None,
                    }
                )
                return output_text
            retry_match = re.search(r"retry in ([\d.]+)s", error_text, re.IGNORECASE)
            if retry_match:
                wait_seconds = max(1, int(float(retry_match.group(1)) + 0.999))
                output_text = (
                    f"⏳ Too many requests right now. Please wait ~{wait_seconds} seconds and try again."
                )
                set_cached_output(cache_key, output_text, ttl_seconds=wait_seconds)
                log_request(
                    {
                        "model": MODEL_NAME,
                        "target_year": target_year,
                        "input_text": user_input,
                        "output_text": output_text,
                        "error_text": error_text,
                        "cache_hit": False,
                        "input_tokens": None,
                        "output_tokens": None,
                        "input_cost_usd": None,
                        "output_cost_usd": None,
                        "total_cost_usd": None,
                    }
                )
                return output_text
            output_text = "⏳ Too many requests right now. Please wait a bit and try again."
            set_cached_output(cache_key, output_text, ttl_seconds=20)
            log_request(
                {
                    "model": MODEL_NAME,
                    "target_year": target_year,
                    "input_text": user_input,
                    "output_text": output_text,
                    "error_text": error_text,
                    "cache_hit": False,
                    "input_tokens": None,
                    "output_tokens": None,
                    "input_cost_usd": None,
                    "output_cost_usd": None,
                    "total_cost_usd": None,
                }
            )
            return output_text
        output_text = f"Error: {error_text}"
        log_request(
            {
                "model": MODEL_NAME,
                "target_year": target_year,
                "input_text": user_input,
                "output_text": output_text,
                "error_text": error_text,
                "cache_hit": False,
                "input_tokens": None,
                "output_tokens": None,
                "input_cost_usd": None,
                "output_cost_usd": None,
                "total_cost_usd": None,
            }
        )
        return output_text

theme = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    primary_hue="indigo",
)

css = """
#white-box {
    background-color: var(--input-background-fill);
    border: var(--input-border-width) solid var(--input-border-color);
    border-radius: var(--input-radius);
    padding: var(--input-padding);
    /* NEW: Setup explicit heights and add a scrollbar for long outputs */
    min-height: 250px; 
    max-height: 350px; 
    overflow-y: auto; 
}
"""

with gr.Blocks(title="Language Rewinder", theme=theme, css=css) as demo:
    gr.Markdown("# ⏪ Language Rewinder")
    gr.Markdown("Adapt your writing for historical accuracy and translate modern slang into the language of the past.")
    
    # NEW: Removed equal_height=True so the columns operate independently
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Modern Phrase",
                placeholder="e.g., No cap, her rizz is actually insane.",
                lines=4,
                max_length=500,
            )
            year_slider = gr.Slider(
                minimum=1900,
                maximum=2025,
                value=1930,
                step=5,
                label="Target Era"
            )
            submit_btn = gr.Button("Adapt to the Past", variant="primary", size="md")

        with gr.Column(scale=1):
            gr.HTML("<div style='display: inline-block; color: var(--block-label-text-color); font-size: var(--block-label-text-size); font-weight: var(--block-label-text-weight); margin-bottom: -10px; margin-left: 0px; padding: 6px 10px; border-radius: 8px; background-color: rgba(99, 202, 241, 0.18); '>Historical Translation</div>")
            output_text = gr.Markdown(elem_id="white-box")

    submit_btn.click(fn=translate_text, inputs=[input_text, year_slider], outputs=output_text)
    input_text.submit(fn=translate_text, inputs=[input_text, year_slider], outputs=output_text)

    gr.Examples(
        examples=[
            ["What's up dude? you chillin'?", 1940],
            ["This startup is looking for a deep dive into our synergy.", 1920],
            ["J'ai trop le seum, le mec m'a ghosté de ouf.", 1990],
        ],
        inputs=[input_text, year_slider],
        label="Try these modern examples"
    )

demo.queue(default_concurrency_limit=5).launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    ssr_mode=False,
)