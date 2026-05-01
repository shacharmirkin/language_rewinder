---
title: Language Rewinder
emoji: 🔁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.14.0"
python_version: "3.11"
app_file: app.py
license: mit
pinned: false
---

# Language Rewinder

Language Rewinder transforms modern writing into historically grounded language for a chosen year or decade.

The app uses *Gemini 2.5 flash* to rewrite modern wording into era-appropriate vocabulary and tone, while avoiding obvious anachronisms for the selected period. In addition to the adapted text it returns a short etymology note explaining key substitutions.

## How it works

You provide:
- a modern sentence or short paragraph
- a target era using a year slider

The app returns:
- a historically adapted version of your text
- a compact note describing notable linguistic changes

## GitHub Repository

Source code: [github repo](https://github.com/shacharmirkin/lang_rewinder)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Disclaimer

This project is a simple AI-assisted linguistic simulation, not a profound linguistic reconstruction tool. Outputs may contain inaccuracies, oversimplifications, or stylistic artifacts from the model. 

⚠️ For observability and cost tracking, request/response text may be logged by the app runtime. Do not submit sensitive personal, private, or regulated data. Account names and other personal data are not logged.
