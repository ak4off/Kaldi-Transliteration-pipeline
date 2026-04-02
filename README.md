# Transliteration Pipeline for Kaldi Text

A robust, production-ready pipeline to transliterate **English and mixed-script tokens** into native Indian languages in **Kaldi-format text files**, with built-in validation and error analysis.

Designed for **ASR post-processing**, multilingual corpora cleaning, and transliteration-heavy pipelines.

---

## Features

* Handles **pure English words**
* Handles **mixed tokens** (e.g. `deliveryலாம்`, `3Gசிம்`)
* Preserves **Kaldi tags** (`<tam>`, `<asr>`, timestamps)
* Detects **unexpected third-language characters**
* Provides **full audit trail** (before/after + leftovers)
* Modular **4-step pipeline** (can run independently)
* Works with multiple Indian languages (Tamil, Hindi, Telugu, etc.)

---

## Pipeline Overview

The pipeline runs in **4 steps**:

### 1. Extract & Analyze

* Extract English words
* Detect mixed tokens (English + native script)
* Identify other-language contamination

Outputs:

```
.english_words
.mixed_tokens
.other_lang_words
.other_lang
```

---

### 2. Transliteration

* Uses AI-based transliteration model
* Converts:

  * English → native script
  * Mixed tokens → partially transliterated tokens

Outputs:

```
.english_words.translit
.english_words.mixed_translit
```

---

### 3. Replacement

* Replaces words in original Kaldi file
* Handles:

  * Word boundaries correctly
  * Mixed tokens safely
  * Optional tag preservation

Output:

```
.translit_replaced
```

---

### 4. Post-check (Critical)

* Finds leftover:

  * English words
  * Mixed tokens
  * Other-language text

Outputs:

```
.translit_replaced.leftover_*
```

---

## Installation

```bash
pip install ai4bharat-transliteration tqdm
```
```
 refer AI4Bharath Xlit[https://pypi.org/project/ai4bharat-transliteration/]
```
---

## Usage

### Run full pipeline

```bash
python translit_pipeline.py --input text --lang ta
```

---

### Preserve ASR tags

```bash
python translit_pipeline.py --input text --lang ta --keep-tags
```

---

### Run specific steps

```bash
# Step 1 only
python translit_pipeline.py --input text --lang ta --steps 1

# Step 2 + 3
python translit_pipeline.py --input text --lang ta --steps 2 3

# Only validation
python translit_pipeline.py --input text --lang ta --steps 4
```

---

## Input Format

Kaldi-style:

```
utt_id text
```

Example:

```
0001 <tam><asr><0.00> deliveryலாம் வந்தது <2.34>
```

---

## Supported Languages

| Code | Language  |
| ---- | --------- |
| ta   | Tamil     |
| hi   | Hindi     |
| te   | Telugu    |
| kn   | Kannada   |
| ml   | Malayalam |
| bn   | Bengali   |
| gu   | Gujarati  |
| pa   | Punjabi   |
| or   | Odia      |
| mr   | Marathi   |

---

## Mixed Token Handling (Key Feature)

Example:

```
deliveryலாம் → டெலிவரிலாம்
சின்னமான3G → சின்னமான3ஜி
```

✔ English part is transliterated
✔ Native part is preserved
✔ Done safely without breaking tokens

---

## Output Guarantees

After Step 4:

* You know exactly:

  * What failed
  * What remains untranslated
  * Where errors are

This makes it **debuggable and production-safe**.

---

## File Outputs Summary

| File                 | Description              |
| -------------------- | ------------------------ |
| `.english_words`     | extracted English tokens |
| `.mixed_tokens`      | tokens mixing scripts    |
| `.translit_replaced` | final output             |
| `.leftover_*`        | error/debug files        |

---

## Code Structure

Main script:


Core components:

* Unicode-based language detection
* Regex-driven token parsing
* AI transliteration (`XlitEngine`)
* Safe replacement engine
* Validation pipeline

---

## Limitations

* Transliteration quality depends on model output
* Does not handle:

  * Semantic translation
  * Context-aware rewriting
* Mixed tokens with complex patterns may need manual review

---

## Best Use Cases

* ASR transcript cleanup
* Multilingual dataset normalization
* Preprocessing for NLP pipelines
* Speech/text alignment pipelines

---

## Future Improvements

* Better handling of:

  * Numerics inside tokens
  * Acronyms (e.g. "IT", "AI")
* Batch inference optimization
* Language detection per token (auto mode)
* Integration with ASR pipelines

---

## Contributing

Pull requests are welcome. If you're working on multilingual ASR/NLP, this pipeline is meant to be extended.

---

##  License

MIT License

---

##  TL;DR

If your dataset looks like:

```
deliveryலாம் வந்தது
```

This pipeline turns it into:

```
டெலிவரிலாம் வந்தது
```

…and tells you exactly what it couldn’t fix.

---
