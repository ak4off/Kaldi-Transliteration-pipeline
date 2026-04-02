#!/usr/bin/env python3
"""
Transliteration Pipeline for Kaldi-format text files.
Transliterates English words → target language script in 4 steps.

Usage:
    # Run all steps
    python translit_pipeline.py --input text --lang ta

    # Run all steps with tag preservation
    python translit_pipeline.py --input text --lang ta --keep-tags

    # Run only specific steps  (loads prior step outputs from disk automatically)
    python translit_pipeline.py --input text --lang ta --steps 1
    python translit_pipeline.py --input text --lang ta --steps 2 3
    python translit_pipeline.py --input text --lang ta --steps 3 4

Steps:
    1. Extract & check    →  .english_words
                             .other_lang_words
                             .other_lang          (full utterances)
                             .mixed_tokens        (NEW — tokens like deliveryலாம்)
    2. Transliterate      →  .english_words.translit
    3. Replace            →  .translit_replaced
    4. Post-check         →  .translit_replaced.leftover_english_words
                             .translit_replaced.leftover_other_words
                             .translit_replaced.leftover_english  (full utterances)
                             .translit_replaced.leftover_other    (full utterances)
                             .translit_replaced.leftover_mixed_tokens (NEW)
"""

import sys
import re
import os
import argparse
from tqdm import tqdm
from ai4bharat.transliteration import XlitEngine as xe


# ─────────────────────────────────────────────────────────────────
# Language Unicode Ranges
# ─────────────────────────────────────────────────────────────────
LANG_UNICODE = {
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "hi": (0x0900, 0x097F),  # Hindi / Devanagari
    "mr": (0x0900, 0x097F),  # Marathi (Devanagari)
    "te": (0x0C00, 0x0C7F),  # Telugu
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "bn": (0x0980, 0x09FF),  # Bengali
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "pa": (0x0A00, 0x0A7F),  # Punjabi / Gurmukhi
    "or": (0x0B00, 0x0B7F),  # Odia
}

TAG_PATTERN          = re.compile(r"<[^>]+>")
ENGLISH_PATTERN      = re.compile(r"[A-Za-z]")
ENGLISH_WORD_PATTERN = re.compile(r"[a-zA-Z]+")
TOKEN_PATTERN        = re.compile(r"\S+")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def get_lang_re(lang: str) -> re.Pattern:
    """Matches any character inside the target language's Unicode block."""
    s, e = LANG_UNICODE[lang]
    return re.compile(rf"[\u{s:04X}-\u{e:04X}]")


def get_other_lang_re(lang: str) -> re.Pattern:
    """
    Matches any character that is:
      - NOT ASCII  (covers English, digits, punctuation, spaces …)
      - NOT in the target language's Unicode block
    i.e. characters from an unexpected third script.
    """
    s, e = LANG_UNICODE[lang]
    return re.compile(rf"[^\x00-\x7F\u{s:04X}-\u{e:04X}]")


def is_mixed_token(token: str, lang_re: re.Pattern) -> bool:
    """
    Returns True when a whitespace-separated token contains BOTH
    ASCII letters (English) AND target-language script characters,
    e.g.  deliveryலாம்  or  சின்னமான3G.
    Tags (<…>) are never mixed tokens.
    """
    if TAG_PATTERN.fullmatch(token):
        return False
    return bool(ENGLISH_PATTERN.search(token)) and bool(lang_re.search(token))


def extract_english_part(token: str) -> str:
    """
    Pull out the contiguous English letter run(s) from a mixed token.
    e.g. 'deliveryலாம்' → 'delivery'
         'சின்னமான3G'   → 'G'   (only alphabetic, not digits)
    Returns all runs joined by space so each can be transliterated.
    """
    return " ".join(ENGLISH_WORD_PATTERN.findall(token))


def extract_native_part(token: str, lang: str) -> str:
    """
    Pull out the target-language characters from a mixed token.
    e.g. 'deliveryலாம்' → 'லாம்'
    """
    s, e = LANG_UNICODE[lang]
    return "".join(ch for ch in token if s <= ord(ch) <= e)


def strip_tags_for_analysis(text: str) -> str:
    """Remove <…> tags so their content is not analysed as language data."""
    return TAG_PATTERN.sub("", text)


def mask_tags(text: str):
    """
    Replace every <…> tag with a null-byte placeholder so the replacement
    regex cannot touch tag content.
    Returns (masked_text, [(placeholder, original_tag), …]).
    """
    placeholders = []
    masked = text
    for i, tag in enumerate(TAG_PATTERN.findall(text)):
        ph = f"\x00TAG{i}\x00"
        masked = masked.replace(tag, ph, 1)
        placeholders.append((ph, tag))
    return masked, placeholders


def restore_tags(text: str, placeholders: list) -> str:
    for ph, tag in placeholders:
        text = text.replace(ph, tag)
    return text


def parse_kaldi_line(line: str):
    """'utt_id rest of text' → (utt_id, text) or (None, None)."""
    parts = line.split(maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (None, None)


def require_file(path: str, description: str):
    """Abort with a clear message if a required intermediate file is missing."""
    if not os.path.exists(path):
        print(f"\n  ✗ Required file not found: {path}")
        print(f"    ({description})")
        print("    Run the earlier step first, or include it in --steps.\n")
        sys.exit(1)


def write_word_list(path: str, words: set):
    with open(path, "w", encoding="utf-8") as f:
        for word in sorted(words):
            f.write(word + "\n")


def extract_other_lang_tokens(text: str, other_lang_re: re.Pattern) -> set:
    """Return all whitespace-separated tokens that contain other-lang chars."""
    return {tok for tok in TOKEN_PATTERN.findall(text) if other_lang_re.search(tok)}


def collect_mixed_tokens(text: str, lang_re: re.Pattern) -> set:
    """Return all tokens that are a mix of English + target-language script."""
    return {tok for tok in TOKEN_PATTERN.findall(text) if is_mixed_token(tok, lang_re)}


# ─────────────────────────────────────────────────────────────────
# Step 1 — Extract word lists & flag other-language utterances
# ─────────────────────────────────────────────────────────────────
def step1_extract_and_check(input_path: str, lang: str, keep_tags: bool) -> dict:
    print("\n[Step 1] Extracting word lists & checking for other-language characters …")

    other_lang_re    = get_other_lang_re(lang)
    lang_re          = get_lang_re(lang)
    english_words    = set()
    other_lang_words = set()
    other_lang_lines = []
    mixed_tokens     = set()   # NEW — tokens that blend English + native script

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue

        utt_id, text = parse_kaldi_line(line)
        if utt_id is None:
            continue

        analysis_text = strip_tags_for_analysis(text) if keep_tags else text

        # ── purely-English words (no native script in the token) ──
        for token in TOKEN_PATTERN.findall(analysis_text):
            if TAG_PATTERN.fullmatch(token):
                continue
            if is_mixed_token(token, lang_re):
                # Mixed token — handle below; do NOT add to english_words yet
                mixed_tokens.add(token)
                # Extract the English portion so it gets transliterated too
                for eng_part in ENGLISH_WORD_PATTERN.findall(token):
                    english_words.add(eng_part)
            elif ENGLISH_PATTERN.search(token) and not lang_re.search(token):
                # Purely-English token
                for word in ENGLISH_WORD_PATTERN.findall(token):
                    english_words.add(word)

        # ── other-language tokens & lines ─────────────────────────
        found = extract_other_lang_tokens(analysis_text, other_lang_re)
        if found:
            other_lang_words.update(found)
            other_lang_lines.append(line)

    english_words_path    = input_path + ".english_words"
    other_lang_words_path = input_path + ".other_lang_words"
    other_lang_path       = input_path + ".other_lang"
    mixed_tokens_path     = input_path + ".mixed_tokens"   # NEW

    write_word_list(english_words_path,    english_words)
    write_word_list(other_lang_words_path, other_lang_words)

    # NEW — write mixed tokens with their English and native parts annotated
    with open(mixed_tokens_path, "w", encoding="utf-8") as f:
        f.write("# mixed_token\tenglish_part\tnative_part\n")
        for tok in sorted(mixed_tokens):
            eng_part = extract_english_part(tok)
            # Native part: everything that is NOT ASCII
            native_part = "".join(ch for ch in tok if ord(ch) > 0x7F)
            f.write(f"{tok}\t{eng_part}\t{native_part}\n")

    with open(other_lang_path, "w", encoding="utf-8") as f:
        for line in other_lang_lines:
            f.write(line + "\n")

    print(f"  ✓ English words         ({len(english_words):>6}):  {english_words_path}")
    print(f"  ✓ Mixed tokens          ({len(mixed_tokens):>6}):  {mixed_tokens_path}")
    print(f"  ✓ Other-lang words      ({len(other_lang_words):>6}):  {other_lang_words_path}")
    print(f"  ✓ Other-lang utterances ({len(other_lang_lines):>6}):  {other_lang_path}")

    return {
        "english_words":      english_words,
        "english_words_path": english_words_path,
        "mixed_tokens":       mixed_tokens,
        "mixed_tokens_path":  mixed_tokens_path,
        "other_lang_words":   other_lang_words,
    }


def load_step1_outputs(input_path: str) -> dict:
    """Load Step 1 outputs from disk when step 1 is not being run."""
    english_words_path = input_path + ".english_words"
    mixed_tokens_path  = input_path + ".mixed_tokens"
    require_file(english_words_path, "run Step 1 to generate the english_words file")

    with open(english_words_path, "r", encoding="utf-8") as f:
        english_words = {ln.strip() for ln in f if ln.strip()}

    mixed_tokens = set()
    if os.path.exists(mixed_tokens_path):
        with open(mixed_tokens_path, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("#") or not ln.strip():
                    continue
                mixed_tokens.add(ln.split("\t")[0])

    print(f"  ✓ Loaded {len(english_words):>6} English words  ← {english_words_path}")
    print(f"  ✓ Loaded {len(mixed_tokens):>6} mixed tokens   ← {mixed_tokens_path}")
    return {
        "english_words":      english_words,
        "english_words_path": english_words_path,
        "mixed_tokens":       mixed_tokens,
        "mixed_tokens_path":  mixed_tokens_path,
    }


# ─────────────────────────────────────────────────────────────────
# Step 2 — Transliterate English → target language
# ─────────────────────────────────────────────────────────────────
def step2_transliterate(english_words_path: str, english_words: set,
                        mixed_tokens: set, lang: str) -> dict:
    print(f"\n[Step 2] Transliterating {len(english_words)} English word(s) → [{lang}] …")

    model = xe(lang, beam_width=10, rescore=True)
    translit_map = {}

    # ── 1. Transliterate purely-English words (unchanged behaviour) ──
    for word in tqdm(sorted(english_words), desc="  Transliterating words"):
        result = model.translit_word(word, lang_code=lang, topk=1)
        translit_map[word] = result[0] if result else word  # fallback: keep original

    # ── 2. Build translit entries for mixed tokens ────────────────
    #    For each mixed token, replace only its English run(s) with their
    #    transliteration and keep the native-script suffix/prefix intact.
    #    e.g.  deliveryலாம்  →  டெலிவரிலாம்
    #    The full token is added to translit_map so Step 3 can do a
    #    single-pass replacement (whole-token match).
    mixed_translit_map = {}   # token → fully-transliterated token
    if mixed_tokens:
        print(f"  ✓ Building translit for {len(mixed_tokens)} mixed token(s) …")
        for tok in sorted(mixed_tokens):
            result_tok = tok
            for eng_part in ENGLISH_WORD_PATTERN.findall(tok):
                translit_form = translit_map.get(eng_part)
                if translit_form is None:
                    res = model.translit_word(eng_part, lang_code=lang, topk=1)
                    translit_form = res[0] if res else eng_part
                    translit_map[eng_part] = translit_form  # cache for future use
                result_tok = result_tok.replace(eng_part, translit_form, 1)
            mixed_translit_map[tok] = result_tok

    # Write combined translit file (words + mixed tokens)
    translit_path = english_words_path + ".translit"
    with open(translit_path, "w", encoding="utf-8") as f:
        # Pure-English words
        for eng, native in sorted(translit_map.items()):
            f.write(f"{eng}\t{native}\n")

    # Write mixed-token translit as a separate log for inspection
    mixed_translit_path = english_words_path + ".mixed_translit"
    with open(mixed_translit_path, "w", encoding="utf-8") as f:
        f.write("# original_token\ttransliterated_token\n")
        for tok, translit in sorted(mixed_translit_map.items()):
            f.write(f"{tok}\t{translit}\n")

    print(f"  ✓ Transliteration map   ({len(translit_map):>6} entries):  {translit_path}")
    print(f"  ✓ Mixed token translit  ({len(mixed_translit_map):>6} entries):  {mixed_translit_path}")

    return {
        "translit_map":       translit_map,
        "translit_path":      translit_path,
        "mixed_translit_map": mixed_translit_map,
    }


def load_step2_outputs(input_path: str) -> dict:
    """Load Step 2 outputs from disk when step 2 is not being run."""
    translit_path       = input_path + ".english_words.translit"
    mixed_translit_path = input_path + ".english_words.mixed_translit"
    require_file(translit_path, "run Step 2 to generate the transliteration map")

    translit_map = {}
    with open(translit_path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) == 2:
                translit_map[parts[0]] = parts[1]

    mixed_translit_map = {}
    if os.path.exists(mixed_translit_path):
        with open(mixed_translit_path, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("#") or not ln.strip():
                    continue
                parts = ln.rstrip("\n").split("\t")
                if len(parts) == 2:
                    mixed_translit_map[parts[0]] = parts[1]

    print(f"  ✓ Loaded {len(translit_map):>6} word entries        ← {translit_path}")
    print(f"  ✓ Loaded {len(mixed_translit_map):>6} mixed-token entries ← {mixed_translit_path}")
    return {
        "translit_map":       translit_map,
        "translit_path":      translit_path,
        "mixed_translit_map": mixed_translit_map,
    }


# ─────────────────────────────────────────────────────────────────
# Step 3 — Replace English words in the Kaldi text file
# ─────────────────────────────────────────────────────────────────
def step3_replace(input_path: str, translit_map: dict,
                  mixed_translit_map: dict, keep_tags: bool) -> dict:
    print("\n[Step 3] Replacing English words (and mixed tokens) in Kaldi text file …")

    if not translit_map and not mixed_translit_map:
        print("  ⚠ Transliteration maps are empty — nothing to replace.")
        return {"replaced_path": input_path}

    # ── Build combined replacement map ───────────────────────────
    # Mixed tokens go in first (longer, more specific patterns take priority)
    combined_map = {}
    combined_map.update(translit_map)
    combined_map.update(mixed_translit_map)   # mixed tokens override plain words if clashing

    # Longest key first to avoid partial-match clobbering (e.g. "IT" before "I")
    sorted_keys  = sorted(combined_map.keys(), key=len, reverse=True)

    # Mixed tokens may contain native script — use a token-boundary-aware pattern
    # \b works for ASCII; for mixed tokens we additionally try a raw token match.
    pure_english_keys = [k for k in sorted_keys if not any(ord(c) > 0x7F for c in k)]
    mixed_keys        = [k for k in sorted_keys if any(ord(c) > 0x7F for c in k)]

    # Pattern for pure-English keys (word boundary safe)
    eng_pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, pure_english_keys)) + r")\b"
    ) if pure_english_keys else None

    # Pattern for mixed tokens — match the whole token (surrounded by space/start/end)
    mixed_pattern = re.compile(
        r"(?<!\S)(" + "|".join(map(re.escape, mixed_keys)) + r")(?!\S)"
    ) if mixed_keys else None

    replaced_path  = input_path + ".translit_replaced"
    replaced_count = 0

    def apply_replacements(text: str) -> str:
        # 1. Replace mixed tokens first (they are more specific)
        if mixed_pattern:
            text = mixed_pattern.sub(lambda m: combined_map[m.group(0)], text)
        # 2. Replace remaining pure-English words
        if eng_pattern:
            text = eng_pattern.sub(lambda m: combined_map[m.group(0)], text)
        return text

    with open(input_path,    "r", encoding="utf-8") as fin, \
         open(replaced_path, "w", encoding="utf-8") as fout:

        for raw in fin:
            line = raw.rstrip("\n")

            if not line.strip():
                fout.write("\n")
                continue

            utt_id, text = parse_kaldi_line(line)
            if utt_id is None:
                fout.write(line + "\n")
                continue

            if keep_tags:
                masked, placeholders = mask_tags(text)
                new_text = restore_tags(apply_replacements(masked), placeholders)
            else:
                new_text = apply_replacements(text)

            if new_text != text:
                replaced_count += 1

            fout.write(f"{utt_id} {new_text}\n")

    print(f"  ✓ Utterances modified   ({replaced_count:>6}):  {replaced_path}")
    return {"replaced_path": replaced_path}


def load_step3_outputs(input_path: str) -> dict:
    """Load Step 3 outputs from disk when step 3 is not being run."""
    replaced_path = input_path + ".translit_replaced"
    require_file(replaced_path, "run Step 3 to generate the replaced file")
    print(f"  ✓ Using existing file  ← {replaced_path}")
    return {"replaced_path": replaced_path}


# ─────────────────────────────────────────────────────────────────
# Step 4 — Post-replacement quality check
# ─────────────────────────────────────────────────────────────────
def step4_final_check(replaced_path: str, lang: str, keep_tags: bool) -> dict:
    print("\n[Step 4] Post-replacement check for residual English / other-lang chars …")

    other_lang_re          = get_other_lang_re(lang)
    lang_re                = get_lang_re(lang)
    leftover_english_words = set()
    leftover_other_words   = set()
    leftover_mixed_tokens  = set()   # NEW
    leftover_english_lines = []
    leftover_other_lines   = []

    with open(replaced_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            utt_id, text = parse_kaldi_line(line)
            if utt_id is None:
                continue

            check_text = strip_tags_for_analysis(text) if keep_tags else text

            # ── Residual mixed tokens ─────────────────────────────
            for tok in TOKEN_PATTERN.findall(check_text):
                if TAG_PATTERN.fullmatch(tok):
                    continue
                if is_mixed_token(tok, lang_re):
                    leftover_mixed_tokens.add(tok)

            # ── Residual English ──────────────────────────────────
            if ENGLISH_PATTERN.search(check_text):
                for word in ENGLISH_WORD_PATTERN.findall(check_text):
                    leftover_english_words.add(word)
                leftover_english_lines.append(line)

            # ── Other language ────────────────────────────────────
            found = extract_other_lang_tokens(check_text, other_lang_re)
            if found:
                leftover_other_words.update(found)
                leftover_other_lines.append(line)

    lef_eng_words_path    = replaced_path + ".leftover_english_words"
    lef_other_words_path  = replaced_path + ".leftover_other_words"
    lef_mixed_tokens_path = replaced_path + ".leftover_mixed_tokens"   # NEW
    lef_eng_path          = replaced_path + ".leftover_english"
    lef_other_path        = replaced_path + ".leftover_other"

    write_word_list(lef_eng_words_path,   leftover_english_words)
    write_word_list(lef_other_words_path, leftover_other_words)

    # NEW — write leftover mixed tokens with annotations
    with open(lef_mixed_tokens_path, "w", encoding="utf-8") as f:
        f.write("# mixed_token\tenglish_part\tnative_part\n")
        for tok in sorted(leftover_mixed_tokens):
            eng_part    = extract_english_part(tok)
            native_part = "".join(ch for ch in tok if ord(ch) > 0x7F)
            f.write(f"{tok}\t{eng_part}\t{native_part}\n")

    with open(lef_eng_path, "w", encoding="utf-8") as f:
        for line in leftover_english_lines:
            f.write(line + "\n")
    with open(lef_other_path, "w", encoding="utf-8") as f:
        for line in leftover_other_lines:
            f.write(line + "\n")

    print(f"  ✓ Leftover English words      ({len(leftover_english_words):>6}):  {lef_eng_words_path}")
    print(f"  ✓ Leftover English utterances ({len(leftover_english_lines):>6}):  {lef_eng_path}")
    print(f"  ✓ Leftover mixed tokens       ({len(leftover_mixed_tokens):>6}):  {lef_mixed_tokens_path}")
    print(f"  ✓ Leftover other-lang words   ({len(leftover_other_words):>6}):  {lef_other_words_path}")
    print(f"  ✓ Leftover other-lang utts    ({len(leftover_other_lines):>6}):  {lef_other_path}")

    return {
        "leftover_english_words": leftover_english_words,
        "leftover_other_words":   leftover_other_words,
        "leftover_mixed_tokens":  leftover_mixed_tokens,
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="4-step English→native transliteration pipeline for Kaldi text files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
examples:
  Run all steps:
    python translit_pipeline.py --input text --lang ta

  Run all steps with tag preservation (<tam>, <asr>, <22.11> …):
    python translit_pipeline.py --input text --lang ta --keep-tags

  Run only step 1  (extract + check):
    python translit_pipeline.py --input text --lang ta --steps 1

  Run steps 2 and 3  (loads step 1 output from disk):
    python translit_pipeline.py --input text --lang ta --steps 2 3

  Run only post-check on an existing replaced file:
    python translit_pipeline.py --input text --lang ta --steps 4
        """,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the Kaldi-format text file  (utt_id<space>text per line).",
    )
    parser.add_argument(
        "--lang", required=True, choices=list(LANG_UNICODE.keys()),
        help="Target language ISO code.  Supported: " + ", ".join(LANG_UNICODE),
    )
    parser.add_argument(
        "--keep-tags", action="store_true",
        help=(
            "Preserve tags like <tam>, <asr>, <22.11> as-is.\n"
            "Tag content is excluded from transliteration AND from\n"
            "all language character checks."
        ),
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, choices=[1, 2, 3, 4],
        metavar="N",
        default=[1, 2, 3, 4],
        help=(
            "Which steps to run (default: 1 2 3 4 — the full pipeline).\n"
            "When a step is omitted its output is loaded from disk.\n"
            "  1  extract english_words / other_lang_words / mixed_tokens\n"
            "  2  transliterate (slow — runs the AI model)\n"
            "  3  replace words in kaldi file\n"
            "  4  post-replacement quality check\n"
            "Example:  --steps 2 3   re-runs replacement with a new translit map"
        ),
    )
    args = parser.parse_args()

    steps = set(args.steps)

    # ── banner ────────────────────────────────────────────────────
    print(f"\n{'═'*56}")
    print(f"  Transliteration Pipeline")
    print(f"  Input  : {args.input}")
    print(f"  Lang   : {args.lang}")
    print(f"  Tags   : {'preserved (--keep-tags)' if args.keep_tags else 'not preserved'}")
    print(f"  Steps  : {' '.join(str(s) for s in sorted(steps))}")
    print(f"{'═'*56}")

    # ── Step 1 ────────────────────────────────────────────────────
    if 1 in steps:
        s1 = step1_extract_and_check(args.input, args.lang, args.keep_tags)
    else:
        print("\n[Step 1] Skipped — loading outputs from disk …")
        s1 = load_step1_outputs(args.input)

    # ── Step 2 ────────────────────────────────────────────────────
    if 2 in steps:
        s2 = step2_transliterate(
            s1["english_words_path"], s1["english_words"],
            s1.get("mixed_tokens", set()), args.lang
        )
    else:
        print("\n[Step 2] Skipped — loading outputs from disk …")
        s2 = load_step2_outputs(args.input)

    # ── Step 3 ────────────────────────────────────────────────────
    if 3 in steps:
        s3 = step3_replace(
            args.input, s2["translit_map"],
            s2.get("mixed_translit_map", {}), args.keep_tags
        )
    else:
        print("\n[Step 3] Skipped — loading outputs from disk …")
        s3 = load_step3_outputs(args.input)

    # ── Step 4 ────────────────────────────────────────────────────
    if 4 in steps:
        step4_final_check(s3["replaced_path"], args.lang, args.keep_tags)
    else:
        print("\n[Step 4] Skipped.")

    # ── summary ───────────────────────────────────────────────────
    print(f"\n{'═'*56}")
    print("  Done!")
    print(f"  Final output : {s3['replaced_path']}")
    print(f"{'═'*56}\n")


if __name__ == "__main__":
    main()