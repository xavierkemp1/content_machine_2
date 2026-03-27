# content_machine_2

## Running locally

**Requirements:** Python 3.11+

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install the package in editable mode
pip install -e .

# 3) Copy the example env file and fill in your keys
cp .env.example .env

# 4) Run the pipeline
python scripts/run_pipeline.py
```

### Required keys in `.env`

| Variable | Purpose |
|---|---|
| `REDDIT_CLIENT_ID` | Reddit API client ID |
| `REDDIT_CLIENT_SECRET` | Reddit API client secret |
| `REDDIT_USER_AGENT` | Reddit API user agent string |
| `TWITTERAPI_IO_KEY` | [twitterapi.io](https://twitterapi.io) API key — used instead of the official X Bearer Token |
| `OPENAI_API_KEY` | OpenAI API key for ranking and rewriting |

Optional model overrides (defaults shown):

```dotenv
OPENAI_RANKING_MODEL=gpt-4.1-mini
OPENAI_REWRITE_MODEL=gpt-4.1-mini
```

#### Piper TTS (optional)

By default the pipeline generates a **silent WAV** as a placeholder for TTS audio.
To enable real speech synthesis via [Piper](https://github.com/rhasspy/piper), set the following variables:

| Variable | Purpose | Default |
|---|---|---|
| `PIPER_EXE` | Path to the `piper` executable | *(unset — uses silent stub)* |
| `PIPER_VOICES_DIR` | Directory containing `.onnx` voice model files | *(required when `PIPER_EXE` is set)* |
| `PIPER_VOICE` | Voice basename (without `.onnx`) | `en_GB-northern_english_male-medium` |

Example (Windows):

```dotenv
PIPER_EXE=C:\Users\YourUsername\piper\piper.exe
PIPER_VOICES_DIR=C:\Users\YourUsername\piper\voices
PIPER_VOICE=en_GB-northern_english_male-medium
```

If `PIPER_EXE` is unset or the file does not exist the pipeline falls back to the silent stub automatically (offline-safe).

Sources (Reddit subreddits, X accounts) and thresholds are configured in `config/sources.json`.

---

The purpose of this project is to automate the process of creating viral content, through the use of various APIs. The project will follow the below flow:

1) Obtaining raw 'potential content'  
Top performing text content will be fetched from Reddit and X using their APIs.  

Post selection will be based on:
- engagement (upvotes / likes / comments)
- recency (e.g. last 24–72 hours)
- subreddit or account relevance (defined in config)
- minimum thresholds (e.g. score > X, comments > Y)

The X accounts and subreddits will be configurable in a control panel config file (e.g. `config/sources.json`).  

Keys and secrets will be stored in a local `.env` file.  

Output of this stage will be a set of normalised text post objects.

---

2) Filtering  

Posts will be filtered using rule-based logic:
- blacklist words/phrases
- duplicate removal (based on similarity or identical titles)
- minimum/maximum length constraints
- removal of posts that rely on images/videos for context

Posts will then be grouped into:
- short (e.g. < 80 words)
- medium (80–200 words)
- long (200+ words)

After filtering, remaining posts will be sent to the OpenAI API and ranked in order of potential virality, based on the following criteria:
- hook strength  
- emotional charge  
- clarity  
- relatability  
- comment bait  
- short-form suitability  

Output from this stage will be the top X posts from each group (short, medium, long), ranked by viral potential.

---

3) Improvements  

Selected posts will be sent to the OpenAI API to improve their suitability for short-form content.

The model should:
- preserve the original meaning and core story
- improve hook strength (first 1–2 lines)
- simplify and tighten wording for narration
- improve pacing for a ~20–45 second delivery

It will return:
- improved hook  
- rewritten narration text  
- title  
- caption  
- hashtags  

Output for this section will be a list of enhanced content objects, ready for TTS.

---

4) Production  

Each enhanced content object will be converted into a short-form video:

- Generate TTS audio from narration text  
- Generate subtitle file (SRT or similar)  
- Select background video (e.g. gameplay loop, stock footage, abstract visuals)  
- Combine audio + subtitles + background into a 9:16 vertical video  

Video rendering will be handled programmatically (e.g. using ffmpeg).

---

5) Export  

Posts will be saved as `.mp4` files in folders based on their content type (theme/niche), for example:

- `/output/aita`
- `/output/confessions`
- `/output/business`
- `/output/dating`

Each video may also include a metadata JSON file containing:
- script
- caption
- hashtags
- source reference

These outputs are intended for manual posting to short-form platforms (TikTok, Instagram Reels, YouTube Shorts).
