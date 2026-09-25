# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
A CPU-only pipeline in three stages: blocking, a pairwise classifier, and a precision-aware
assignment step. Records are normalised with country-agnostic rules. These romanise native-script
names, canonicalise legal forms and address abbreviations, and strip junk such as PO boxes,
placeholders and domain suffixes. A per-country IDF-weighted sparse token index retrieves the top
candidates under name, address and combined views. A LightGBM model scores every candidate pair
on ~75 vectorised similarity and competition features. Each Source 2/3 record is then assigned to
at most one Source 1 entity, using a threshold tuned directly for macro F0.5.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the training split (2.21M S1, 5.03M S2, 5.29M S3 records):
- **Link structure:** 7.64M true links, 3.46 matches per S1 on average (max 11), 5.6% singletons.
  **No S2/S3 record is linked to more than one S1 entity**, so assignment can be exclusive.
  About 25% of S2/S3 records are unmatched distractors.
- **Country:** all true pairs share the same country label, so blocking per country is safe.
  Country is still treated as an open label set (France appears only in the test set).
- **Name noise:** legal suffix dropped, abbreviated or moved to the front ("Pvt. EFS Print
  Ventures Ltd."); word-order swaps; typos; injected accents ("Léarning"); junk prefixes (`--`,
  `<<`); website forms ("oncologyphysicians.com"); trade names with no token overlap ("Excellent &
  Co" ↔ "Orbiaria"). About 23% of Indian S2/S3 names are in native scripts (Devanagari, Tamil,
  Telugu, Kannada, Gujarati, Bengali).
- **Address noise:** component reordering, upper case, abbreviations, "Street"→"Saint",
  truncated house numbers (1631→163), city typos, `PO BOX`, `<NULL>`, `##`, native-script state
  names, landmarks ("Near Sayaji Hotel"). About 3.3% of S2/S3 addresses are empty. US S1 addresses
  carry no ZIP code.
- **Token overlap among true pairs:** 99.8% share a name token or at least 2 address tokens.
  This motivated token-based retrieval.

### 2.2 Solution Strategy
**Approach Type:** Blocking + gradient-boosted pairwise classifier + constrained assignment  
**Core Innovation:** Scalable IDF-weighted sparse retrieval (three views, per country) plus
"target-side competition" features computed against the *full* S1 pool. This lets a model trained
on a 200k-entity sample see the same competition as inference on the full test set. The decision
threshold is tuned directly on out-of-fold macro F0.5, singletons included.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** namespaced tokens per record. These are name tokens, a compact
  concatenated name (to catch domain-style names) and address tokens (words and numbers). They are
  IDF-weighted and L2-normalised into three views: name, address, and name+address. Tokens seen
  once are dropped; tokens with document frequency above 20,000 get zero retrieval weight. For each
  S1 record and each target source, the top-k records under every view are retrieved with a
  multithreaded sparse top-n matrix product (`sparse_dot_topn`, Apache-2.0). The union is capped
  per (S1, source) by the best view score. Blocking runs per country label.
- **Candidate pairs generated:** [fill in: test total; ~N per S1]
- **How you ensured true matches were not lost:** three complementary views, so a match survives
  a missing address (name view) or a trade-name / native-script name (address view). Recall@k was
  measured on held-out training entities for every view and cap: [fill in pair recall].

---

## 4. Matching Model

**Features used:**
- Name features: rapidfuzz ratio / partial / token-sort / token-set / Jaro-Winkler on the core
  name (legal forms removed); compact-name ratio (domain names); IDF-weighted token Jaccard,
  coverage, max shared IDF and unshared IDF mass; legal-form bitmask agreement and conflict;
  first-token match; token counts and length ratio; native-script flag; name frequency in each
  pool (to handle chains).
- Address features: fuzzy ratios on the cleaned address and its alphabetic part; IDF-weighted
  address-token overlap; shared / conflicting numbers; first-number equality and prefix match
  (truncated house numbers); postal-code equality; landmark similarity; missing-address flag.
- Other: retrieval cosines (name / address / combined); rank and gap to the best candidate among
  the S1's candidates (per source and overall); number of candidates; for the target record, its
  best and second-best combined score against all S1 records and this pair's gap to that best
  (target-side competition).

**Model type:** LightGBM binary classifier (MIT licence), 4-fold GroupKFold by S1 entity,
early stopping; final model refit on the full training sample.  
**Threshold selection method:** each target record is kept only for its highest-probability S1.
The probability cut-off is then grid-searched on out-of-fold predictions to maximise macro F0.5
(singletons included), taking the middle of the best plateau for robustness.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [fill in OOF score; per country]
- **Unseen-country check:** leave-one-country-out (train US → test India and vice versa): [fill in]
- **Common false positives (wrong merges):** [fill in]
- **Common false negatives (missed matches):** [fill in]

---

## 6. Conclusion
[fill in]

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: all source in `src/`, entry point `src/pipeline.py`
(`all` = train + predict), which writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. `README.md` has the exact commands and `requirements.txt` the pinned
versions. No external data, APIs or pretrained models are used; the only learned model is LightGBM.

### B. Additional Results
[fill in]

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
