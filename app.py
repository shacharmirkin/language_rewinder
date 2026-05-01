import os
import gradio as gr
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """
You are a Historical Linguist and Translator. Your goal is to rewrite modern text into the vocabulary, 
syntax, and slang of a specific year or decade. 

RULES:
1. ANCHRONISM FILTER: Strictly avoid words, concepts, or technologies that did not exist in the target year.
2. CULTURAL VIBE: Adopt the social tone of the era (e.g., the earnestness of the 40s, the groove of the 70s).
3. EXPLANATION: After the translation, provide a short 'Etymology Note' explaining why you replaced certain modern words.
4. LANGUAGE TRANSFER: If the input is not in English, translate it into the target year's equivalent within that same language. Do not translate between languages (e.g., French stays French).
"""

def translate_text(user_input, target_year):
    if not api_key:
        return "Error: API key not found. Please set GEMINI_API_KEY."
    if not user_input.strip():
        return ""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            ),
            contents=f"Target Year: {target_year}\nText: {user_input}"
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

theme = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    primary_hue="indigo",
)

with gr.Blocks(title="Language Rewinder") as demo:
    gr.Markdown("# 🕰️ Language Rewinder")
    gr.Markdown("Audit your writing for historical accuracy and translate modern slang into the language of the past.")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Group():
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
            with gr.Group():
                output_text = gr.Textbox(
                    label="Historical Translation",
                    value="The translation and etymology notes will appear here.",
                    lines=8,
                    interactive=False,
                )

    submit_btn.click(fn=translate_text, inputs=[input_text, year_slider], outputs=output_text)
    input_text.submit(fn=translate_text, inputs=[input_text, year_slider], outputs=output_text)

demo.queue(default_concurrency_limit=5).launch(theme=theme)