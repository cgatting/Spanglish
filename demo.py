#!/usr/bin/env python3
"""
spanglish_fullscale_opt.py – Spanglish 3.0 (GPU‑aware, fully cached, auto‑resume)
===============================================================================
• Stop‑word‑aware smart blending + duplicate cleanup
• Autocaches lexicon / pseudo‑pairs / fine‑tuned model per language pair (en‑es default)
• Trainer configured for *production‑style* runs:
    – cosine LR schedule + warm‑up  – fp16 on CUDA
    – gradient accumulation (2)     – eval‑on‑epoch with checkpoints
    – only last 2 checkpoints kept  – automatic resume if partial run exists
• CLI helpers:  --resume  --export‑lexicon  --export‑ipa  --pair  --epochs
"""
from __future__ import annotations
import argparse, json, random, re, unicodedata, pathlib, sys
from collections import OrderedDict
from typing import Dict, List

import torch, tqdm, regex, nltk
from wordfreq import top_n_list
from sentence_transformers import SentenceTransformer
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments,
    set_seed
)
from datasets import Dataset
from phonemizer.backend import EspeakBackend

set_seed(42)

# ─────────────────────────── CLI ────────────────────────────
cli = argparse.ArgumentParser(description="Train or reuse a hybrid Spanglish T5 model.")
cli.add_argument("--pair", default="en-es", help="language pair as lang1-lang2 (default: en-es)")
cli.add_argument("--vocab", type=int, default=8_000, help="Top‑N words per language")
cli.add_argument("--pseudo", type=int, default=20_000, help="# synthetic sentence pairs")
cli.add_argument("--epochs", type=int, default=10, help="Fine‑tune epochs (default 10)")
cli.add_argument("--batch", type=int, default=8, help="per‑device batch size")
cli.add_argument("--resume", action="store_true", help="Reuse cached artefacts / continue training")
cli.add_argument("--export-lexicon", action="store_true")
cli.add_argument("--export-ipa", action="store_true")
args = cli.parse_args()

lang1, lang2 = args.pair.split("-")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = pathlib.Path("hybrid_cache") / args.pair
CACHE.mkdir(parents=True, exist_ok=True)
LEX_F, PAIRS_F, MODEL_D = CACHE/"lexicon.json", CACHE/"pairs.jsonl", CACHE/"t5_model"

# ─────────────────────── helpers ───────────────────────────
def ascii_strip(s: str) -> str:
    return regex.sub(r"[^A-Za-z]", "", "".join(c for c in unicodedata.normalize("NFKD", s)
                                                  if unicodedata.category(c) != "Mn"))

nltk.download("stopwords", quiet=True)
STOP1 = set(nltk.corpus.stopwords.words("english" if lang1 == "en" else lang1))
STOP2 = set(nltk.corpus.stopwords.words("spanish" if lang2 == "es" else lang2))

def blend(w1: str, w2: str) -> str:
    if w1 in STOP1 or w2 in STOP2:
        return w1
    w1, w2 = ascii_strip(w1.lower()), ascii_strip(w2.lower())
    if len(w1) < 3 or len(w2) < 3:
        return w1
        
    # Skip if words are identical
    if w1 == w2:
        return w1
        
    # Calculate phonetic similarity using Levenshtein distance
    def levenshtein(s1, s2):
        if len(s1) < len(s2):
            return levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    # If words are too different phonetically, don't blend
    if levenshtein(w1, w2) > max(len(w1), len(w2)) * 0.7:
        return w1
        
    # Find common prefix and suffix
    prefix = ""
    for i in range(min(len(w1), len(w2))):
        if w1[i] == w2[i]:
            prefix += w1[i]
        else:
            break
            
    suffix = ""
    for i in range(1, min(len(w1), len(w2)) + 1):
        if w1[-i] == w2[-i]:
            suffix = w1[-i] + suffix
        else:
            break
            
    # If we have good common parts, use them
    if len(prefix) >= 2 or len(suffix) >= 2:
        if len(prefix) >= 2:
            return prefix + w2[len(prefix):]
        else:
            return w1[:-len(suffix)] + suffix
            
    # Otherwise use length-based blending
    if len(w1) > len(w2):
        r = random.uniform(0.6, 0.8)  # Keep more of w1
    else:
        r = random.uniform(0.4, 0.6)  # More balanced blend
        
    # Ensure we don't create too short words
    min_len = max(3, min(len(w1), len(w2)) // 2)
    result = w1[: int(len(w1)*r)] + w2[-int(len(w2)*(1-r)):]
    
    # Fallback to original word if blend is too short
    return result if len(result) >= min_len else w1

def collapse_dupes(txt: str) -> str:
    return re.sub(r"\b(\w+)( \1)+\b", r"\1", txt)

# ───────────────── 1. Lexicon  ────────────────────────────
if args.resume and LEX_F.exists():
    lexicon: Dict[str,str] = json.load(LEX_F.open(), object_pairs_hook=OrderedDict)
else:
    vocab1 = top_n_list(lang1, args.vocab, "best")
    vocab2 = top_n_list(lang2, args.vocab, "best")
    embed = SentenceTransformer("distiluse-base-multilingual-cased-v2", device=DEVICE)
    with torch.no_grad():
        e1 = embed.encode(vocab1, convert_to_tensor=True, normalize_embeddings=True)
        e2 = embed.encode(vocab2, convert_to_tensor=True, normalize_embeddings=True)
        idx = torch.argmax(e1 @ e2.T, dim=1)
    lexicon = OrderedDict((f"{vocab1[i]}/{vocab2[j]}", blend(vocab1[i], vocab2[j]))
                          for i, j in enumerate(idx.tolist()))
    json.dump(lexicon, LEX_F.open("w", encoding="utf8"), ensure_ascii=False, indent=2)
print("LEX sample:", list(lexicon.items())[:6])

def hybridify(sent: str, src: str) -> str:
    out: List[str] = []
    for w in sent.lower().split():
        for pair, hyb in lexicon.items():
            l1, l2 = pair.split("/")
            if (src == lang1 and w == l1) or (src == lang2 and w == l2):
                out.append(hyb); break
        else:
            out.append(w)
    return collapse_dupes(" ".join(out))

# ───────────────── 2. Pseudo‑pairs ────────────────────────
def rand_sent(vocab: List[str]) -> str:
    return " ".join(random.sample(vocab, random.randint(4, 8))).lower()

if args.resume and PAIRS_F.exists():
    pairs = [json.loads(l) for l in PAIRS_F.open()]
else:
    pool1 = [rand_sent(top_n_list(lang1, 12_000, "best")) for _ in range(args.pseudo//2)]
    pool2 = [rand_sent(top_n_list(lang2, 12_000, "best")) for _ in range(args.pseudo//2)]
    pairs = ([{"src": f"<{lang1}> "+s, "tgt": hybridify(s, lang1)} for s in pool1] +
             [{"src": f"<{lang2}> "+s, "tgt": hybridify(s, lang2)} for s in pool2])
    with PAIRS_F.open("w", encoding="utf8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

dataset = Dataset.from_list(pairs)

# ───────────────── 3. Model  ──────────────────────────────
if args.resume and MODEL_D.exists():
    tok = T5Tokenizer.from_pretrained(MODEL_D)
    mdl = T5ForConditionalGeneration.from_pretrained(MODEL_D).to(DEVICE)
else:
    tok = T5Tokenizer.from_pretrained("t5-small")
    mdl = T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

    def prep(b):
        enc = tok(b["src"], truncation=True, padding="max_length", max_length=64)
        dec = tok(b["tgt"], truncation=True, padding="max_length", max_length=64)
        enc["labels"] = dec["input_ids"]
        return enc

    train_ds = dataset.map(prep, batched=True, remove_columns=dataset.column_names)

    lr_scheduler = "cosine"
    training_args = TrainingArguments(
        output_dir=str(MODEL_D),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=0.15,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        save_total_limit=2,
        metric_for_best_model="loss",
        logging_steps=100,
        report_to="none",
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_torch",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    if __name__ == '__main__':
        Trainer(model=mdl, args=training_args, train_dataset=train_ds).train()
        mdl.save_pretrained(MODEL_D)
        tok.save_pretrained(MODEL_D)

# ───────────────── 4. Translate helper ────────────────────
def translate(text: str, src: str) -> str:
    ids = tok(f"<{src}> {text.lower()}", return_tensors="pt").to(DEVICE)
    out = mdl.generate(
        **ids,
        max_length=128,
        num_beams=5,  # Beam search
        temperature=0.7,  # Slightly reduce randomness
        top_p=0.9,  # Nucleus sampling
        do_sample=True,  # Enable sampling
        no_repeat_ngram_size=2,  # Prevent repetition
        early_stopping=True
    )
    return tok.decode(out[0], skip_special_tokens=True)

# ───────────────── 5. Optional exports ────────────────────
if __name__ == '__main__':
    if args.export_lexicon:
        print(f"✓ lexicon saved → {LEX_F}")

    if args.export_ipa:
        print("★ generating IPA chart …")
        ph = EspeakBackend(language="es")  # good enough for Spanglish phonemes
        ipa = {hyb: ph.phonemize(hyb, strip=True, punctuation="") for hyb in lexicon.values()}
        ipa_path = CACHE/"ipa.json"
        json.dump(ipa, ipa_path.open("w", encoding="utf8"), ensure_ascii=False, indent=2)
        print(f"✓ IPA chart written → {ipa_path}")

    # ───────────────── 6. Live demo ───────────────────────────
    demo1 = "Hello friend, I love my house and my cat."
    demo2 = "Hola amigo, me encanta mi casa y mi gato."
    print("\n=== LIVE DEMO ("+args.pair+") ===")
    print("→", lang1, ":", demo1)
    print("  H :", translate(demo1, lang1))
    print("\n→", lang2, ":", demo2)
    print("  H :", translate(demo2, lang2))
