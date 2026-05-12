"""
drug_bias.py — Vocabulary bias toward prescription/drug names.

Post-processes any inference backend's output: each predicted word is
matched against a drug-name dictionary; if the closest match is within
a small Levenshtein distance, the prediction is replaced with the
canonical drug name.

Why this helps prescription OCR specifically:
- Drug names use non-English orthography (Hydrochlorothiazide, Atorvastatin)
- General-purpose models / language models "correct" them to wrong words
- The set of clinically relevant drug names is closed (~3000 in common
  use; ~200 cover >80 % of US prescriptions), so fuzzy match against the
  closed set is a tractable problem with a known-correct target

Drug list below is the top ~200 most-prescribed drugs in the US (2023),
publicly tracked by ClinCalc from federal Medicare prescribing data.
Expand by editing _DRUG_NAMES_RAW or by writing your own list to
data/drug_names.txt (one name per line).
"""

import os


# Top ~200 most-prescribed drugs in the US. Generic names only — brand
# names would inflate the list and most prescriptions are written generic.
_DRUG_NAMES_RAW = """
Atorvastatin Levothyroxine Lisinopril Metformin Amlodipine Metoprolol Albuterol
Omeprazole Losartan Gabapentin Hydrochlorothiazide Sertraline Simvastatin
Acetaminophen Montelukast Furosemide Pantoprazole Escitalopram Rosuvastatin
Bupropion Citalopram Tramadol Trazodone Fluoxetine Tamsulosin Prednisone
Carvedilol Meloxicam Duloxetine Clopidogrel Pravastatin Cyclobenzaprine
Insulin Amoxicillin Cephalexin Azithromycin Ciprofloxacin Doxycycline
Hydroxyzine Alprazolam Lorazepam Clonazepam Diazepam Zolpidem Buspirone
Venlafaxine Mirtazapine Paroxetine Lamotrigine Topiramate Pregabalin Lacosamide
Levetiracetam Carbamazepine Valproate Quetiapine Risperidone Aripiprazole
Olanzapine Lithium Methylphenidate Amphetamine Adderall Ritalin Strattera
Atomoxetine Guanfacine Clonidine Hydralazine Spironolactone Eplerenone
Triamterene Bumetanide Torsemide Indapamide Chlorthalidone Atenolol Bisoprolol
Nebivolol Propranolol Diltiazem Verapamil Nifedipine Felodipine Isradipine
Ramipril Enalapril Benazepril Quinapril Captopril Fosinopril Trandolapril
Valsartan Irbesartan Olmesartan Telmisartan Candesartan Eprosartan Azilsartan
Warfarin Apixaban Rivaroxaban Dabigatran Edoxaban Heparin Enoxaparin
Aspirin Ibuprofen Naproxen Diclofenac Celecoxib Indomethacin Ketorolac
Morphine Oxycodone Hydrocodone Codeine Fentanyl Hydromorphone Methadone
Buprenorphine Naloxone Naltrexone Tapentadol Tizanidine Baclofen Carisoprodol
Methocarbamol Orphenadrine Diazepam Cyclobenzaprine Metaxalone Dantrolene
Famotidine Ranitidine Cimetidine Esomeprazole Lansoprazole Rabeprazole
Sucralfate Misoprostol Mesalamine Sulfasalazine Loperamide Ondansetron
Promethazine Metoclopramide Prochlorperazine Hyoscyamine Dicyclomine
Glipizide Glyburide Glimepiride Pioglitazone Sitagliptin Saxagliptin Linagliptin
Empagliflozin Dapagliflozin Canagliflozin Liraglutide Semaglutide Dulaglutide
Exenatide Glargine Detemir Lispro Aspart Regular NPH Mixtard
Allopurinol Febuxostat Colchicine Probenecid Methotrexate Hydroxychloroquine
Sulfasalazine Leflunomide Adalimumab Etanercept Infliximab Rituximab
Levocetirizine Cetirizine Loratadine Fexofenadine Diphenhydramine Hydroxyzine
Chlorpheniramine Brompheniramine Doxylamine Meclizine Dimenhydrinate
Fluticasone Mometasone Budesonide Beclomethasone Triamcinolone Dexamethasone
Prednisolone Methylprednisolone Hydrocortisone Albuterol Levalbuterol Salmeterol
Formoterol Tiotropium Ipratropium Umeclidinium Vilanterol Olodaterol
Sildenafil Tadalafil Vardenafil Avanafil Finasteride Dutasteride Tamsulosin
Silodosin Alfuzosin Doxazosin Terazosin Prazosin Solifenacin Tolterodine
Oxybutynin Mirabegron Darifenacin Trospium Estrogen Progesterone Testosterone
Ethinyl Norethindrone Levonorgestrel Drospirenone Norgestimate Desogestrel
Tretinoin Adapalene Clindamycin Erythromycin Mupirocin Ketoconazole Terbinafine
Nystatin Acyclovir Valacyclovir Famciclovir Oseltamivir Zanamivir Baloxavir
Tacrolimus Cyclosporine Mycophenolate Sirolimus Everolimus Azathioprine
mg mcg mg/mL mg/dL g IU mEq tab tabs cap caps qd bid tid qid prn
po pr im iv sc sq sl od os ou stat hs ac pc
""".split()


def _load_drug_names():
    """Load the drug name list. Allows user to override via data/drug_names.txt."""
    names = list(_DRUG_NAMES_RAW)
    custom_path = os.path.join("data", "drug_names.txt")
    if os.path.exists(custom_path):
        with open(custom_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    names.append(line)
    return names


_DRUGS = _load_drug_names()
_DRUG_LOOKUP = {d.lower(): d for d in _DRUGS}


def _bounded_levenshtein(a, b, max_dist):
    """Levenshtein with early termination once distance exceeds max_dist."""
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if not a:
        return len(b) if len(b) <= max_dist else max_dist + 1
    if not b:
        return len(a) if len(a) <= max_dist else max_dist + 1

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        if min(curr) > max_dist:
            return max_dist + 1
        prev = curr
    return prev[-1]


def bias_word(word, max_edit_distance=2):
    """Return the canonical drug name if `word` matches one within edit
    distance `max_edit_distance`; otherwise return `word` unchanged.

    Words shorter than 4 characters skip the fuzzy match — too short for
    edit-distance to be meaningful (everything is 1-2 edits from everything).
    Trailing punctuation is stripped before matching and re-applied after.
    """
    if not word:
        return word

    # Preserve leading/trailing punctuation around the matched core
    stripped = word.rstrip(",.;:!?\"')")
    trailing = word[len(stripped):]
    core = stripped.lstrip("(\"'")
    leading = stripped[:len(stripped) - len(core)]

    if len(core) < 4:
        return word

    core_lower = core.lower()
    if core_lower in _DRUG_LOOKUP:
        return leading + _DRUG_LOOKUP[core_lower] + trailing

    # Skip short dictionary entries (mg, tab, qd, etc.) as fuzzy-match
    # targets — they're too short for edit distance to be meaningful and
    # produce spurious matches like "10mg" → "mg". They still pass through
    # the exact-match branch above when they appear standalone.
    best_drug, best_dist = None, max_edit_distance + 1
    for drug_lower, drug in _DRUG_LOOKUP.items():
        if len(drug_lower) < 5:
            continue
        d = _bounded_levenshtein(core_lower, drug_lower, max_edit_distance)
        if d < best_dist:
            best_dist, best_drug = d, drug

    return (leading + best_drug + trailing) if best_drug else word


def bias_text(text, max_edit_distance=2):
    """Apply word-level drug-name biasing to a multi-word string."""
    if not text:
        return text
    return " ".join(bias_word(w, max_edit_distance) for w in text.split())


def bias_result(result, max_edit_distance=2):
    """Apply drug bias in-place to a backend result dict (text/words/lines).

    Mutates and returns `result` for convenience. Safe to call on results
    from any backend (CRNN, TrOCR, Claude) — only modifies present keys.
    """
    if "text" in result and isinstance(result["text"], str):
        result["text"] = bias_text(result["text"], max_edit_distance)
    for key in ("words", "lines"):
        for item in result.get(key, []) or []:
            if "text" in item:
                item["text"] = bias_text(item["text"], max_edit_distance)
    result["drug_bias"] = True
    return result
