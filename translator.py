import os
import json
import re

try:
    from google import genai
except Exception:
    genai = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None


LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Arabic": "ar",
    "Chinese": "zh-CN",
    "Italian": "it",
    "Korean": "ko",
    "Russian": "ru",
    "Portuguese": "pt",
}


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

gemini_client = None


def init_gemini_client():
    global gemini_client, GEMINI_API_KEY, GEMINI_MODEL
    if gemini_client is not None:
        return gemini_client

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    if not os.getenv("GEMINI_API_KEY"):
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_file):
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip().strip("'\"")
            except Exception:
                pass

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if genai and GEMINI_API_KEY:
        try:
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            print("Gemini translation client initialized.")
            return gemini_client
        except Exception as e:
            print(f"Gemini initialization error: {e}")
            gemini_client = None
    return None


init_gemini_client()



# ============================================================
# NATURAL DUBBING PROMPT
# ============================================================

def build_translation_prompt(texts, target_language):
    numbered_texts = "\n".join(
        f"{i + 1}. {text}"
        for i, text in enumerate(texts)
    )

    return f"""
You are a professional dialogue translator for a video dubbing system.

Translate the following spoken dialogue into natural conversational
{target_language}.

IMPORTANT:

1. Preserve the exact meaning of the original dialogue.

2. Make the translation sound like something a REAL native speaker
   would say in everyday conversation.

3. DO NOT produce textbook, overly formal, literary, or robotic
   translations.

4. Natural code-switching is allowed and encouraged when appropriate.
   If native speakers commonly use an English word while speaking
   {target_language}, KEEP that English word instead of forcing a
   translated equivalent.

5. For Hindi specifically, natural Hinglish is preferred when appropriate.

   Example:
   English:
   "She is my crush and I love her."

   Natural Hindi:
   "वो मेरा crush है और मैं उससे प्यार करता हूँ।"

   NOT:
   "वह मेरा आकर्षण है और मैं उससे प्रेम करता हूँ।"

6. Other examples of natural conversational Hindi:

   "I'll call you later."
   -> "मैं तुम्हें बाद में call करूँगा।"

   "Let's make a plan."
   -> "चलो एक plan बनाते हैं।"

   "I have a meeting tomorrow."
   -> "मेरी कल एक meeting है।"

7. Do NOT insert English words randomly.
   Use English only when it sounds natural for the target language.

8. Preserve:
   - emotions
   - personality
   - slang
   - casual speech
   - humor
   - relationships
   - names
   - brands
   - technical terms

9. Do not add explanations.

10. Do not remove important information.

11. Keep the translation reasonably close in length to the original
    because the translated dialogue will later be used for video dubbing.

12. Translate each numbered segment independently, but use the surrounding
    segments to understand conversational context.

13. Return ONLY a valid JSON array.

The output must contain exactly one translated string for every input segment.

Example output:

[
    "वो मेरा crush है और मैं उससे प्यार करता हूँ।",
    "लेकिन मैंने अभी तक उसे बताया नहीं है।"
]

TARGET LANGUAGE:
{target_language}

DIALOGUE:

{numbered_texts}
"""


# ============================================================
# GEMINI JSON EXTRACTION
# ============================================================

def extract_json_array(response_text):
    """
    Extract a JSON array from Gemini's response.
    Handles cases where Gemini accidentally wraps JSON in ```json ...```.
    """

    if not response_text:
        return None

    text = response_text.strip()

    # Remove markdown code fences
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try direct JSON parsing
    try:
        data = json.loads(text)

        if isinstance(data, list):
            return data

    except json.JSONDecodeError:
        pass

    # Try to find the JSON array inside the response
    match = re.search(r"\[.*\]", text, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(0))

            if isinstance(data, list):
                return data

        except json.JSONDecodeError:
            pass

    return None


# ============================================================
# GEMINI TRANSLATION
# ============================================================

def gemini_translate_texts(texts, target_language="Hindi"):
    """
    Translate multiple transcript segments using Gemini.

    Returns:
        list[str] | None
    """

    if not texts:
        return []

    if not gemini_client:
        init_gemini_client()

    if not gemini_client:
        print("Gemini client unavailable.")
        return None

    prompt = build_translation_prompt(
        texts,
        target_language
    )

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            }
        )

        response_text = response.text

        translations = extract_json_array(response_text)

        if not translations:
            print("Gemini returned invalid translation format.")
            return None

        if len(translations) != len(texts):
            print(
                f"Gemini returned {len(translations)} translations "
                f"for {len(texts)} segments."
            )
            return None

        # Make sure everything is a string
        translations = [
            str(item).strip()
            for item in translations
        ]

        return translations

    except Exception as e:
        print(f"Gemini translation error: {e}")
        return None


# ============================================================
# FALLBACK TRANSLATION
# ============================================================

def fallback_translate_text(text, target_language="Hindi"):

    if not text:
        return ""

    if not GoogleTranslator:
        return text

    try:

        code = LANGUAGE_CODES.get(
            target_language,
            target_language.lower()
        )

        translated = GoogleTranslator(
            source="auto",
            target=code
        ).translate(text)

        return translated if translated else text

    except Exception as e:

        print(
            f"Fallback translation error "
            f"({target_language}): {e}"
        )

        return text


# ============================================================
# SINGLE TEXT TRANSLATION
# ============================================================

def translate_text(text, target_language="Hindi"):

    if not text:
        return ""

    target_key = (target_language or "Hindi").strip()

    if target_key.lower() in {"", "original"}:
        return text

    # Try Gemini first
    gemini_result = gemini_translate_texts(
        [text],
        target_language=target_key
    )

    if gemini_result and len(gemini_result) == 1:

        translated = gemini_result[0]

        if translated:
            return translated

    # Gemini failed -> fallback
    print("Using GoogleTranslator fallback.")

    return fallback_translate_text(
        text,
        target_language=target_key
    )


# ============================================================
# BATCH TRANSLATION
# ============================================================

def batch_translate_texts(
    texts,
    target_language="Hindi",
    batch_size=20
):
    """
    Translate transcript segments in batches using Gemini.

    Gemini receives several segments together so it can understand
    conversational context.

    batch_size prevents excessively large prompts.
    """

    if not texts:
        return []

    target_key = (target_language or "Hindi").strip()

    if target_key.lower() in {"", "original"}:
        return texts

    results = []

    # Process only non-empty strings
    normalized_texts = [
        t.strip() if t else ""
        for t in texts
    ]

    for batch_start in range(
        0,
        len(normalized_texts),
        batch_size
    ):

        batch = normalized_texts[
            batch_start:batch_start + batch_size
        ]

        # Keep empty segments
        non_empty_indices = [
            i for i, text in enumerate(batch)
            if text
        ]

        if not non_empty_indices:
            results.extend([""] * len(batch))
            continue

        non_empty_texts = [
            batch[i]
            for i in non_empty_indices
        ]

        # ----------------------------------------------------
        # Gemini
        # ----------------------------------------------------

        translated_batch = gemini_translate_texts(
            non_empty_texts,
            target_language=target_key
        )

        # ----------------------------------------------------
        # Gemini failed -> fallback
        # ----------------------------------------------------

        if not translated_batch:

            print(
                f"Gemini batch failed. "
                f"Using fallback translation for batch "
                f"{batch_start // batch_size + 1}."
            )

            translated_batch = [
                fallback_translate_text(
                    text,
                    target_language=target_key
                )
                for text in non_empty_texts
            ]

        # ----------------------------------------------------
        # Restore empty segments
        # ----------------------------------------------------

        batch_results = [""] * len(batch)

        for index, translation in zip(
            non_empty_indices,
            translated_batch
        ):
            batch_results[index] = (
                translation or batch[index]
            )

        results.extend(batch_results)

    return results


# ============================================================
# TRANSLATE TRANSCRIPT SEGMENTS
# ============================================================

def translate_segments(
    segments,
    target_language="Hindi"
):
    """
    Translate transcript segments while preserving
    original timing information.
    """

    if not segments:
        return []

    orig_texts = [
        seg.get("text", "").strip()
        for seg in segments
    ]

    translated_texts = batch_translate_texts(
        orig_texts,
        target_language=target_language
    )

    translated_segments = []

    for seg, translated_txt in zip(
        segments,
        translated_texts
    ):

        text = seg.get("text", "").strip()

        if not text:
            continue

        start = float(
            seg.get("start", 0.0)
        )

        end = float(
            seg.get("end", 0.0)
        )

        translated_segments.append({

            "start": start,

            "end": end,

            "duration": max(
                0.1,
                end - start
            ),

            "orig_text": text,

            # Keep this field if your existing
            # dubbing pipeline expects hindi_text.
            "hindi_text": (
                translated_txt or text
            ),

            # More generic field for future languages.
            "translated_text": (
                translated_txt or text
            ),

            "target_language": target_language,
        })

    return translated_segments