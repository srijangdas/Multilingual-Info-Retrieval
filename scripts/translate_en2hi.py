"""Translate MIRACL English evaluation queries into Hindi and preserve query-ID links.

Run from the repository root:
    python scripts/translate_en2hi.py

The generated set uses IDs prefixed with ``en2hi_``. The mapping file links those IDs
back to the English IDs used by eval_qrels.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import requests


ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/miracl_eval/en/eval_queries.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/miracl_eval/en2hi")
    args = parser.parse_args()

    source = json.loads(args.source.read_text(encoding="utf-8"))
    english_to_hindi = {}
    hindi_to_english = {}
    translated = {}
    def translate(pair):
        english_id, english_text = pair
        for attempt in range(3):
            try:
                response = requests.get(
                    "https://translate.googleapis.com/translate_a/single",
                    params={"client": "gtx", "sl": "en", "tl": "hi", "dt": "t", "q": english_text},
                    timeout=30,
                )
                response.raise_for_status()
                segments = response.json()[0]
                hindi_text = "".join(segment[0] for segment in segments if segment and segment[0])
                return english_id, hindi_text.strip()
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(translate, source.items())
        for english_id, hindi_text in results:
            time.sleep(0.1)
            hindi_id = f"en2hi_{english_id}"
            translated[hindi_id] = hindi_text
            english_to_hindi[str(english_id)] = hindi_id
            hindi_to_english[hindi_id] = str(english_id)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "eval_queries.json").write_text(
        json.dumps(translated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mapping = {
        "translation_method": "Google Translate English to Hindi",
        "source": "../en/eval_queries.json",
        "qrels_source": "../en/eval_qrels.json",
        "english_to_hindi": english_to_hindi,
        "hindi_to_english": hindi_to_english,
    }
    (args.output / "query_id_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Translated {len(translated)} queries into {args.output}")


if __name__ == "__main__":
    main()
