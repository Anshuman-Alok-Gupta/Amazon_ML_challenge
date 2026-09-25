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
candidates under name, address and combined views. A small stage-1 LightGBM re-ranker keeps the
best 6 candidates per (S1, source). A stage-2 LightGBM model scores each remaining pair on 66
vectorised similarity and competition features. Each Source 2/3 record is then assigned to at most
one Source 1 entity, using a threshold tuned directly for macro F0.5. Trained on the full training
data, it reaches an **out-of-fold macro F0.5 of 0.9739** (US 0.9797, India 0.9653), with 97.3%
candidate-pair recall. It runs end to end in about 1.5 h on an 8-vCPU / 30 GB machine.

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
on a 400k-entity sample see the same competition as inference on the full test set. The decision
threshold is tuned directly on out-of-fold macro F0.5, singletons included.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** namespaced tokens per record. These are name tokens, a compact
  concatenated name (to catch domain-style names) and address tokens (words and numbers). They are
  IDF-weighted and L2-normalised into three views: name, address, and name+address. Tokens seen
  once are dropped; tokens with document frequency above 20,000 get zero retrieval weight. For each
  S1 record and each target source, the top-k records under every view are retrieved with a
  multithreaded sparse top-n matrix product (`sparse_dot_topn`, Apache-2.0). The union is capped
  per (S1, source) by the best view score. Blocking runs per country label, split further by
  region (state) with a residual block for records whose region is missing or unrecognised.
- **Region handling (data-driven, identical for every country label):** normalisation detects a
  region only from an explicit region name or code. Three structural rules keep that detection
  honest, and none of them names a specific state or country:
  - A segment counts as a region only if its letters are a region and its digits (if any) look
    like a postal code. "Fl 0" (a floor) and ".../Hr/16" (a house number) are no longer read as
    Florida or Haryana.
  - The last short word of a segment is promoted to a region code only in the address's final
    segment or next to a postal code. "Oak Ct" now becomes "oak court" instead of Connecticut.
  - When region-only segments disagree ("Oor, …, West Bengal"; "Washington, DC" vs "DC,
    Washington"), all candidates are kept. Each country then picks the candidate most common among
    its own unambiguous S1 records, so both orderings resolve to the same block.

  Per country, a region holding under 0.05% of that country's S1 records is treated as noise and
  sent to the residual block. On true training pairs, the share of pairs that land in *different*
  region blocks (and so can never be matched) fell from 0.089% to 0.031% in India and from 0.349%
  to 0.063% in the US.
- **Region inference for region-less records (`regions.py`):** some sources write only a city or
  a sub-region (a département, a county), so the record gets no region and falls into the
  residual block, where it competes with every S1 of the country. For each country, the pipeline
  learns from its own records which tail-of-address words co-occur with one region: at least 20
  records and at least 98% purity. It then assigns that region to region-less records whose
  witnesses all agree, in two passes. No labels, place-name lists or external data are used.
  - Validation, done by hiding known regions: 99.65% accurate in India and 99.43% in the US.
  - On the test set, region-less French S2/S3 records fell from **66.7% to 6.2%**. Before this
    fix, two-thirds of France went into one mixed residual block, with 23 candidates per S1
    against 13 elsewhere.
- **Vocabulary fixes:** doubled letters are collapsed before dictionary lookup, which had made
  every hand-written key containing a double letter unreachable, among them `poona→pune`,
  `puducherry`, `null` and `estt`. The keys and values are now collapsed the same way
  automatically.
  - Honorifics (`shri`, `sri`, `smt`, `mr`, `dr`, a leading "M/s") and romanised native-script
    legal forms (`pra`, `praibhet`, `limird`; `li` only after another legal form) were added.
  - Each one was mined from true pairs of the stage-1 sample only (`mine_vocab.py`, support ≥ 50),
    never from the entities the OOF score is measured on.
- **Transliteration-robust key:** a `k|` token holds the consonant skeleton of the whole compact
  name, so "sevnkeyr" (romanised "सेवन केयर") and "sevencare" meet. The compact-name token is now
  emitted for one-word names too (domain-style names such as "glistensons.com").
- **Stage-1 re-ranking:** the retrieved pool (about 112 targets per S1) is scored by a light
  LightGBM model (300 rounds, trained on a 60k-S1 sample disjoint from the matcher's training
  sample). It keeps the top 6 per (S1, source) with probability ≥ 0.002. The candidate set it
  produces is exactly what the matcher scores, and it is what `candidate_pairs.tsv` contains.
- **Candidate pairs generated:**
  - Test: **24,653,408 pairs for 1,732,544 S1 entities (14.2 per S1)**, drawn from a retrieval
    pool of 203M pairs. By country: France 259,452 S1, India 809,986 S1, US 663,106 S1. Only 7 S1
    entities have no candidate.
  - Training sample: 5,100,284 pairs for 400,000 S1 (12.75 per S1), 26.4% of them positive.
  - Reduction ratio versus the full cross product within each country: > 0.99999.
- **How you ensured true matches were not lost:** three complementary views, so a match survives
  a missing address (name view) or a trade-name / native-script name (address view). Recall was
  measured on held-out training entities:
  - Retrieval pool recall: **98.44%** (sample A, 60k S1).
  - Final candidate-pair recall after re-ranking: **97.27%** (sample B, 400k S1).
  - This recall caps the reachable macro F0.5 at **0.990**.

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

- Transliteration and alias features: ratio and Jaro-Winkler on the consonant skeletons of the
  full compact names; IDF overlap of the whole-name skeleton token; and the best token-set score
  when a "formerly known as / DBA / trading as" alternate name is present on either side.

**Model type:** LightGBM binary classifier (MIT licence), 4-fold GroupKFold by S1 entity,
early stopping; final model refit on the full training sample.  
**Threshold selection method:** each target record is kept only for its highest-probability S1.
Two decoders are then compared on out-of-fold predictions, and the one with the higher macro F0.5
is kept:
- a global probability cut-off;
- per-S1 expected-F0.5 decoding. For each S1 it keeps the top-k candidates that maximise
  1.25·Σpᵢ(top k) / (0.25·Σpᵢ(all) + k), and it predicts an empty list when ∏(1 − pᵢ) is higher.

Each decoder has exactly one tuned number, grid-searched for macro F0.5 (singletons included),
taking the middle of the best plateau for robustness.

**Guarding against overfitting:**
- Every change is accepted only on a paired bootstrap of per-entity OOF F0.5 against the
  previous model (`--baseline`). The lower 95% bound of the gain must be above zero.
- No country-specific thresholds, models or features are used.
- Vocabulary is mined only from entities outside the scored sample.
- The public leaderboard is not used for tuning.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** out of fold, 4-fold GroupKFold over a 400k-S1 training sample, with
  exclusive assignment and threshold 0.74:

  | Scope | S1 entities | OOF macro F0.5 |
  |---|---|---|
  | **Overall** | 400,000 | **0.9739** |
  | US | 239,845 | 0.9797 |
  | India | 160,155 | 0.9653 |

  The development run (15% of regions, 60k sample) scored 0.966 overall (US 0.974, India 0.956).
  Using the full data added about 0.008. The best boosting iterations per fold were stable
  (1009–1208), and the final model was refit with 1189 rounds.
- **Unseen-country check:** the leave-one-country-out check (`pipeline.py validate`) was not
  re-run for the full-data model. Indirect evidence that the model transfers to France, which
  has no labels: the France test set gets a match rate and match count per entity close to the
  labelled countries.

  | Test country | S1 entities | S1 with ≥ 1 match | Avg matches per matched S1 |
  |---|---|---|---|
  | France | 259,452 | 94.6% | 3.28 |
  | India | 809,986 | 93.9% | 3.25 |
  | US | 663,106 | 94.2% | 3.36 |

  Overall, 1,630,703 of the 1,732,544 test S1 entities (94.1%) have at least one match, and there
  are 5,712,533 matched pairs. In training, 94.4% of S1 entities have at least one true link.
- **Error mix** (OOF errors for 20,000 S1 entities): 4,480 wrong pairs, of which only 318 (7%) are
  false positives and 4,162 (93%) are false negatives. This is the intended trade-off for F0.5.
- **Common false positives (wrong merges):** the model is confident on these (median probability
  0.89). 53% have a near-identical name (token-set similarity ≥ 90) and 41% a near-identical
  address.
  - *Near-duplicate distractors.* The same or a typo'd name at almost the same address, with one
    house or flat number changed: "171 Ram Vihar" ↔ "17 Ram Vihar", "8/303 Eastend Apartments" ↔
    "8/307 …", "H. No. 975" ↔ "H.NO 984", "Clifford Fidelity" ↔ "CLIFFORD FÍDEIIIYT" with no
    target address.
  - *Native-script names that romanise to a different business* at a matching address (8% of
    false positives).
- **Common false negatives (missed matches):**
  - *Lost in blocking* (1,839, 44% of false negatives; 68% of them in India). 32% have a
    native-script target name ("भारत इंफोटेक एलएलपी", "आनंद ऑल मैनेजमेंट लिमिटेड"). 47% have a
    name with little token overlap: trade names ("Ali's Cafe" ↔ "Fayedelta") or truncated names.
    20% have an empty target address, which removes the address view's help.
  - *Scored but below the threshold* (2,323, 56%; median probability 0.29). 56% have an empty or
    placeholder target address, even though 60% have a near-identical name. Without an address,
    the model can't rule out a same-name distractor, so it stays conservative. The other cases are
    legal-form changes ("P.C." → "LLC"), IDs or phone numbers appended to the name ("Optimal
    Financial (ID: 18587)", "Mccrary LP Center - 4615162495"), and renumbered street numbers.

---

## 6. Conclusion
Precision-first blocking plus a gradient-boosted matcher gives an out-of-fold macro F0.5 of 0.974
on the full training data. It runs on CPU only, uses no external data, and takes about 1.5 h end to
end on 8 vCPUs. The features that mattered most were the stage-1 score, the target-side
competition features (`t_gap_best`, `t_is_best`, `t_margin`), address token-set similarity and
number agreement. These allow exclusive, threshold-tuned assignment to reject near-duplicate
distractors. The main remaining losses, in order of size:
1. Records with a missing address that are scored below the threshold.
2. Native-script and trade-name pairs lost in blocking. Blocking recall caps F0.5 at 0.990.
3. A small number of number-perturbed near-duplicates that get through.

Better transliteration for retrieval, name-rarity features for address-less pairs, and stricter
house-number conflict features are the clearest next steps.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: all source in `src/`, entry point `src/pipeline.py`
(`all` = train + predict), which writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. `README.md` has the exact commands and `requirements.txt` the pinned
versions. No external data, APIs or pretrained models are used; the only learned model is LightGBM.

### B. Additional Results
Full run on AWS EC2 `m7i.2xlarge` (8 vCPU, 30 GB RAM), started with `bash aws/run_on_ec2.sh`
(`pipeline.py train --sample 400000`, then `pipeline.py predict`, then the official validator).

| Stage | Detail | Time |
|---|---|---|
| Normalise train | 2.21M S1, 5.03M S2, 5.29M S3 rows | 193 s |
| Stage-1 pools + fit | 60k S1 sample A; pool recall 98.44% | 305 s |
| Stage-2 CV | 5.10M pairs, 66 features, 4 folds | 768 s |
| Refit | 1189 rounds | 180 s |
| Normalise test | 1.73M S1, 4.89M S2, 5.08M S3 rows | 192 s |
| Test blocking | 24.65M candidate pairs | about 31 min |

Top features by gain: `stage1`, `t_gap_best`, `addr_tset`, `t_is_best`, `num_jac`, `t_margin`,
`first_num_prefix`, `name_idf_cov_t`, `legal_equal`, `num_cov_s1`, `legal_conflict`,
`name_idf_unshared`, `core_partial`, `skel_idf_cov_t`, `clean_tset`.

Validator (`utils/validate_submission.py`, also run with `--check-ids`): **PASS**. It found every
one of the 1,732,544 S1 entities in both files: `matching_results.tsv` has 1,630,703 non-empty
rows and `candidate_pairs.tsv` has 1,732,537 non-empty rows.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
