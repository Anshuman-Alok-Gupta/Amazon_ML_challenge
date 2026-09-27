# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
The pipeline has three stages: blocking, a stacked pairwise matcher, and a precision-aware
assignment step.
- **Normalisation:** country-agnostic rules romanise native-script names, canonicalise legal forms
  and address abbreviations, and strip junk such as PO boxes, placeholders and domain suffixes.
- **Blocking:** a per-country, per-region IDF-weighted sparse token index retrieves a deep pool of
  targets under name, address and combined views.
- **Stage-1 re-ranker:** a LightGBM model re-ranks that pool using cosine scores plus cheap
  name / house-number evidence, and keeps the best 8 candidates per (S1, source).
- **Matcher:** a two-level stack.
  - Level 1: LightGBM and XGBoost on 72 similarity and competition features, plus the logit of a
    fine-tuned multilingual transformer cross-encoder (`intfloat/multilingual-e5-small`, MIT,
    118M parameters) that reads the raw text of both records.
  - Level 2: a LightGBM meta-model that adds relational features describing each S1's other
    candidates.
- **Assignment:** each Source 2/3 record goes to at most one Source 1 entity, and a decoder tuned
  directly for macro F0.5 cuts the lists.

**Results:** trained on the full training data, it reaches an **out-of-fold macro F0.5 of 0.9840**
(US 0.9864, India 0.9804), with 98.25% candidate-pair recall, and 0.975 on the public
leaderboard. Retrained at the test set's distractor density (18.7% of training S1 entities hidden,
see Section 5), it scores **0.9836 out of fold on the harder frame and 0.976 on the public
leaderboard**. Everything runs on CPU: 8 vCPU / 30 GB, EC2 m7i.2xlarge.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the training split (2.21M S1, 5.03M S2, 5.29M S3 records):
- **Link structure:** 7.64M true links, 3.46 matches per S1 on average (max 11), 5.6% singletons.
  **No S2/S3 record is linked to more than one S1 entity**, so assignment can be exclusive.
  About 26% of S2/S3 records are unmatched distractors.
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
- **Train/test shift:** the test set has about 24% more S2/S3 records per S1 than training data.
  - Per country: India 5.82 vs 4.68, US 5.76 vs 4.67; France has 5.53.
  - Predicted matches per S1 are the same in both splits (about 3.4), so the extra test records
    are unmatched distractors.
  - This is the main reason the leaderboard sits below the out-of-fold score (Section 5).

### 2.2 Solution Strategy
**Approach Type:** Blocking + learned re-ranking + stacked matcher (gradient-boosted trees and a
transformer cross-encoder) + constrained assignment  
**Core Innovations:**
- Scalable IDF-weighted sparse retrieval (three views, per country and region).
- "Target-side competition" features computed against the *full* S1 pool. These let a model
  trained on a 400k-entity sample see the same competition as inference on the full test set.
- A cross-encoder that compares both records token by token across scripts.
- Relational features that let the meta-model reason about an S1's candidates jointly: the
  near-identical "twin" distractor, and the best match in the other source.
- Decision rules tuned directly on out-of-fold macro F0.5, singletons included.
- Every component is kept only if a paired bootstrap shows a gain.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** namespaced tokens per record.
  - Token types: name tokens, a compact concatenated name (to catch domain-style names), and
    address tokens (words and numbers).
  - They are IDF-weighted and L2-normalised into three views: name, address, and name+address.
    Tokens seen once are dropped; tokens with document frequency above 20,000 get zero
    retrieval weight.
  - For each S1 record and each target source, the top-k records under every view are retrieved
    with a multithreaded sparse top-n matrix product (`sparse_dot_topn`, Apache-2.0).
  - Blocking runs per country label, split further by region (state), with a residual block for
    records whose region is missing or unrecognised.
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
  residual block, where it competes with every S1 of the country.
  - For each country, the pipeline learns from its own records which tail-of-address words
    co-occur with one region: at least 20 records and at least 98% purity.
  - It then assigns that region to region-less records whose witnesses all agree, in two passes.
    No labels, place-name lists or external data are used.
  - Validation, done by hiding known regions: 99.65% accurate in India and 99.43% in the US.
  - On the test set, region-less French S2/S3 records fell from **66.7% to 6.2%**.
- **Vocabulary fixes:** doubled letters are collapsed before dictionary lookup, which had made
  every hand-written key containing a double letter unreachable, among them `poona→pune`,
  `puducherry`, `null` and `estt`. The keys and values are now collapsed the same way
  automatically.
  - Honorifics (`shri`, `sri`, `smt`, `mr`, `dr`, a leading "M/s") and romanised native-script
    legal forms (`pra`, `praibhet`, `limird`; `li` only after another legal form) were added.
  - Each one was mined from true pairs of the stage-1 sample only (`mine_vocab.py`, support ≥ 50),
    never from the entities the OOF score is measured on.
- **Transliteration-robust key:** a `k|` token holds the consonant skeleton of the whole compact
  name, so "sevnkeyr" (romanised "सेवन केयर") and "sevencare" meet. The compact-name token is
  emitted for one-word names too (domain-style names such as "glistensons.com").
- **Stage-1 re-ranking (learned):** the retrieved pool (about 116 targets per S1) is scored by a
  LightGBM model (300 rounds), trained on a 60k-S1 sample disjoint from the matcher's training
  sample.
  - Beyond the three cosine scores and their within-pool ranks, it uses cheap pair evidence:
    - name token-set similarity (rapidfuzz)
    - IDF coverage of the name on both sides, and whole-name skeleton overlap
    - shared house numbers (count and Jaccard), and whether the first numbers are equal
    - whether the target name is in a native script, and whether its address is empty
  - This evidence tells a same-address record with a garbled or native-script name, or a changed
    house number, apart from the many same-name decoys of a region.
  - It keeps the top 8 per (S1, source) with probability ≥ 0.005. The candidate set it produces
    is exactly what the matcher scores, and it is what `candidate_pairs.tsv` contains.
  - Compared with the earlier re-ranker (cosine features only, top 6), it raised candidate-pair
    recall from 97.86% to **98.25%** while cutting candidates by **39%** (12.4 → 7.7 per S1).
- **Candidate pairs generated:**
  - Test: **15,814,996 pairs for 1,732,544 S1 entities (9.1 per S1)**. By country: France 259,452
    S1, India 809,986 S1, US 663,106 S1. Only 41 S1 entities have no candidate.
  - Training sample: 3,061,330 pairs for 400,000 S1 (7.65 per S1), 44.4% of them positive.
  - Reduction ratio versus the full cross product within each country: > 0.99999.
- **How you ensured true matches were not lost:** three complementary views, so a match survives
  a missing address (name view) or a trade-name / native-script name (address view). Recall was
  measured on held-out training entities:
  - Retrieval pool recall: **98.92%** (sample A, 60k S1).
  - Final candidate-pair recall after re-ranking: **98.25%** (sample B, 400k S1).
  - This recall caps the reachable macro F0.5 at **0.9944**.

---

## 4. Matching Model

**Level-0 features (72, all country-agnostic, vectorised):**
- Name features: rapidfuzz ratio / partial / token-sort / token-set / Jaro-Winkler on the core
  name (legal forms removed); compact-name ratio (domain names); IDF-weighted token Jaccard,
  coverage, max shared IDF and unshared IDF mass; legal-form bitmask agreement and conflict;
  first-token match; token counts and length ratio; native-script flag; name frequency in each
  pool (to handle chains).
- Address features: fuzzy ratios on the cleaned address and its alphabetic part; IDF-weighted
  address-token overlap; shared / conflicting numbers; first-number equality and prefix match
  (truncated house numbers); postal-code equality; landmark similarity; missing-address flag.
- Transliteration and alias features: ratio and Jaro-Winkler on the consonant skeletons of the
  full compact names; IDF overlap of the whole-name skeleton token; and the best token-set score
  when a "formerly known as / DBA / trading as" alternate name is present on either side.
- Other: retrieval cosines (name / address / combined) and the stage-1 score; rank and gap to
  the best candidate among the S1's candidates (per source and overall); number of candidates;
  and, for the target record, its best and second-best combined score against all S1 records
  and this pair's gap to that best (target-side competition).

**Cross-encoder (`src/cross_encoder.py`):**
- `intfloat/multilingual-e5-small` (MIT, 118M parameters), fine-tuned as a one-logit pair
  classifier on the raw text `"<name> | <address>"` of both records (max 128 tokens).
- It sees native scripts, French and house numbers directly, which the hand-made similarity
  scores only approximate.
- Training:
  - 300k candidate pairs of a third training sample C (150k S1, disjoint from the stage-1 and
    matcher samples), so its scores on the matcher's training sample are out-of-sample.
  - 1 epoch, AdamW (lr 5e-5, linear warm-up), batch 64, bfloat16 on the CPU's AMX units.
  - The 250k-token word-embedding matrix is frozen, to save memory and keep the pretrained
    multilingual vocabulary intact.
  - Held-out pairs: **AUC 0.9972**, log-loss 0.066, accuracy 97.6%.
- On CPU it scores only the pairs whose level-0 probability lies in [0.05, 0.95]: 174k training
  pairs and 1.21M test pairs, at about 270 pairs/s. The rest get a missing value, which the tree
  models handle natively.

**Stack (`src/stack.py`), all levels on the same 4 GroupKFold folds (by S1 entity):**
- Level 1: LightGBM (MIT) and XGBoost (Apache-2.0) on the level-0 features plus the
  cross-encoder logit.
- Level 2: LightGBM on the level-0 features, the level-1 logits, and 21 relational features. The
  relational features are built from the level-1 probabilities of the *same S1's other*
  candidates:
  - rank, gap to the best and number of likely candidates, within the S1 and within the source;
  - **twin**: the best other candidate of the same source (its probability, name / address
    similarity, and whether its house number matches the S1). A near-identical sibling with a
    higher probability is the typical false merge.
  - **anchor**: the S1's best candidate in the *other* source (its probability and similarity to
    this candidate). A true S2 match usually looks like the true S3 match, which rescues garbled
    names and empty addresses.

  Target-side relational features are deliberately left out: the training frame is an S1 sample,
  so they would not match the full test set.
- A third relational round was tried and rejected by the bootstrap.

**Threshold selection method:** each target record is kept only for its highest-probability S1.
Two decoders are then compared on out-of-fold predictions, and the one with the higher macro F0.5
is kept:
- a global probability cut-off;
- per-S1 expected-F0.5 decoding. For each S1 it keeps the top-k candidates that maximise
  1.25·Σpᵢ(top k) / (0.25·Σpᵢ(all) + k), and it predicts an empty list when ∏(1 − pᵢ) is higher.

Each decoder has exactly one tuned number, grid-searched for macro F0.5 (singletons included),
taking the middle of the best plateau for robustness. The final model uses expected-F0.5
decoding with floor 0.45.

**Guarding against overfitting:**
- Every change and every stacking level is accepted only on a paired bootstrap of per-entity OOF
  F0.5 against the previous model. The lower 95% bound of the gain must be above zero, and no
  country's mean gain may be negative.
- No country-specific thresholds, models or features are used.
- The stage-1 re-ranker, the cross-encoder and the matcher are trained on three disjoint S1
  samples. Vocabulary is mined only from entities outside the scored sample.
- The public leaderboard is not used for tuning.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), out of fold:** 4-fold GroupKFold over a 400k-S1 training sample, with
  exclusive assignment. The ablation below shows what each component added; every step is a
  bootstrap KEEP.

  | Model | OOF macro F0.5 | Gain over previous row (95% CI) |
  |---|---|---|
  | Previous round (LightGBM, cosine-only re-ranker) | 0.9779 | – |
  | + learned stage-1 re-ranker v2 | 0.9792 | +0.0012 [+0.0011, +0.0014] |
  | + cross-encoder logit (level-1 LightGBM) | 0.9832 | +0.0040 [+0.0038, +0.0042] |
  | + XGBoost (level-1 mean) | 0.9833 | +0.0001 [+0.0000, +0.0001] |
  | **+ level-2 relational meta-model (final)** | **0.9840** | +0.0008 [+0.0007, +0.0009] |

  Final model by country:

  | Scope | S1 entities | OOF macro F0.5 |
  |---|---|---|
  | **Overall** | 400,000 | **0.9840** |
  | US | 239,845 | 0.9864 |
  | India | 160,155 | 0.9804 |

  Without the cross-encoder, the same stack scores 0.9815 (US 0.9847, India 0.9768).
- **Leaderboard vs out-of-fold:**

  | Submission | OOF | Public leaderboard |
  |---|---|---|
  | First round | 0.9739 | 0.958 |
  | Normalisation round | 0.9779 | 0.964 |
  | Stack without cross-encoder | 0.9815 | 0.967 |
  | Stack with cross-encoder | 0.9840 | 0.975 |
  | **Same stack, trained at test distractor density (final)** | **0.9836** | **0.976** |

  - The gap between OOF and leaderboard stayed roughly constant (1.4–1.6 points) while both
    rose, until the cross-encoder narrowed it to 0.9. A constant gap points to a systematic
    train/test difference rather than overfitting.
  - The difference found is distractor density (Section 2.1): about 42% of test S2/S3 records
    are unmatched, against 26% in training.
  - The model's own uncertainty rises accordingly. The share of S1 entities with a candidate at
    probability 0.2–0.9 is 30% on India test against 19% in training, 19% on US test against
    16%, and 32% on France.
  - The cross-encoder, which compares the raw text of both records, is the most robust to those
    extra near-duplicates. It gained more on the leaderboard (+0.008) than out of fold (+0.0025).
- **Unseen country (France):** there are no labels. France gets a match rate and match count per
  entity close to the labelled countries:

  | Test country | S1 entities | S1 with ≥ 1 match | Avg matches per S1 |
  |---|---|---|---|
  | France | 259,452 | 94.7% | 3.36 |
  | India | 809,986 | 94.0% | 3.30 |
  | US | 663,106 | 94.3% | 3.38 |

  Overall, 1,632,486 of the 1,732,544 test S1 entities (94.2%) have at least one match, and there
  are 5,788,266 matched pairs (final submission). In training, 94.4% of S1 entities have at least one true link.
- **Error analysis** (level-0 model, OOF errors for 20,000 S1 entities; about 93% of wrong pairs
  are false negatives, the intended trade-off for F0.5):
  - **Common false positives (wrong merges):** near-duplicate distractors. These have the same or
    a typo'd name at almost the same address, with one house or flat number changed: "1003
    Anmoltower" ↔ "1005 Anmoltower", "3300 Commonwealth Drive" ↔ "3301 …", "171 Ram Vihar" ↔ "17
    Ram Vihar". A dropped legal form is another pattern ("Black Technology Private Limited" ↔
    "Black Technology Limited"). The relational twin features and the cross-encoder target
    exactly these cases.
  - **Common false negatives (missed matches):**
    - *Lost in blocking:* native-script target names ("బ్రైట్ ఇన్ఫోటెక్ ప్రైవేట్ లిమిటెడ్" ↔
      "Bright Infotech Private Limited"), and random or garbled target names whose address alone
      matches ("QUOSOLVEO, 3615 Diane Ln" ↔ "Gomez Pioneer Co, 3617 Diane Lane"). The learned
      stage-1 re-ranker recovered part of these (recall 97.86% → 98.25%).
    - *Scored below the cut-off:* more than half have an empty or placeholder target address.
      Without an address, the model can't rule out a same-name distractor, so it stays
      conservative. Other cases are legal-form changes, IDs or phone numbers appended to the
      name, and renumbered street numbers.

---

## 6. Conclusion
A learned blocking stage plus a stacked matcher gives an out-of-fold macro F0.5 of 0.9840 and a
public leaderboard score of 0.975, raised to 0.976 by training at the test set's distractor density. Everything runs on CPU, and the only external artefact is a
small MIT-licensed pretrained encoder. The largest single gain came from the transformer
cross-encoder (+0.004 OOF, +0.008 on the leaderboard). The next largest came from relational
features that judge each candidate against the S1's other candidates, and from a stage-1
re-ranker that uses cheap name and house-number evidence.

The remaining losses, in order of size:
1. Train/test distractor density. Test has about 24% more unmatched S2/S3 records per S1. The
   final model is trained under the same density: `train --hide-s1 auto` hides a fraction
   h = 1 − (train targets per S1) / (test targets per S1) = 1 − 4.677 / 5.754 = 0.187 of the
   training S1 entities (413,344 of 2,206,821) from the whole pipeline, so their S2/S3 records
   become unmatched distractors, as on test. h is measured from the row counts, the same for
   every country. The hidden entities are drawn outside samples A, B and C, so the stage-1 model
   and the cross-encoder are reused unchanged. On this harder frame the stack scores 0.9836 OOF
   (US 0.9859, India 0.9800) and 0.976 on the leaderboard.
2. Records with a missing address that are scored below the cut-off.
3. Native-script and trade-name pairs lost in blocking; the candidate set caps F0.5 at 0.9944.
4. A cross-encoder limited by CPU: trained on 300k pairs and applied only to uncertain pairs.
   A GPU would allow training on all 1.15M sample-C pairs and scoring every candidate.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: all source is in `src/`, and the entry point is
`src/pipeline.py`.
- Level 0: `all` (train + predict).
- Cross-encoder: `ce-train` and `ce-score`.
- Stack: `stack` and `stack-predict`, which writes `output/matching_results.tsv` and
  `output/candidate_pairs.tsv`.
- `train` and `predict` cache their model inputs under `cache/frames/`, so the later commands
  never regenerate candidates.
- `README.md` has the exact commands and `requirements.txt` the pinned versions.
- Learned models: LightGBM (MIT), XGBoost (Apache-2.0), and the fine-tuned
  `intfloat/multilingual-e5-small` (MIT, 118M parameters; pretrained weights from the Hugging Face
  hub). No external data, APIs or lookups are used.

### B. Additional Results
Full run on AWS EC2 `m7i.2xlarge` (8 vCPU, 30 GB RAM, no GPU), driven by `aws/run_on_ec2.sh`.
Some steps ran concurrently.

| Stage | Detail | Time |
|---|---|---|
| Normalise train | 2.21M S1, 5.03M S2, 5.29M S3 rows | 235 s |
| Stage-1 pools + fit | 60k S1 sample A; pool recall 98.92% | 811 s |
| Candidates + features | samples B (400k) and C (150k) | 1,239 s |
| Level-0 CV + refit | 3.06M pairs, 72 features, 4 folds | 657 s |
| Test blocking + features | 15.81M candidate pairs (shared the CPU with cross-encoder training) | 3,082 s |
| Cross-encoder training | 300k pairs, 1 epoch, 4 threads | 140 min |
| Cross-encoder scoring | 174k train + 1.21M test pairs | 89 min |
| Stack (levels 1–3, ablation) | 3.06M pairs | 34 min |
| Stack prediction | 15.81M test pairs | about 12 min |

Top level-0 features by gain: `stage1`, `addr_tset`, `t_gap_best`, `comb_cos`, `t_margin`,
`legal_conflict`, `name_idf_cov_t`, `legal_equal`, `name_idf_unshared`, `first_num_prefix`,
`t_is_best`, `first_num_equal`.

Validator (`utils/validate_submission.py`, run with `--check-ids`): **PASS**. It found every
one of the 1,732,544 S1 entities in both files: `matching_results.tsv` has 1,632,486 non-empty
rows and `candidate_pairs.tsv` has 1,732,503 non-empty rows (final submission).

### C. Rejected Experiment: Graph Neural Network
The candidate pairs form a bipartite graph (S1 entities ↔ S2/S3 records). A GNN can read what
the other candidates of the same S1, and the other S1 entities claiming the same record, say
about a pair. It was tried and **not used**, because it lost on the leaderboard. The code is on
the repository's `gnn` branch.

- **Training graph.** The stack's training frame is an S1 sample (~18%), so most competing S1
  entities of a record are missing from it. The GNN was therefore trained on a separate frame:
  every S1 entity of a seeded 15% of the training regions (193k S1, 1.54M pairs), with the same
  18.7% of S1 hidden as in the final model. The stack was refit without those S1 entities, so
  its scores on the graph frame are out-of-sample.
- **Model.** Edge states (the level-2 input and logit, plus node degrees) are passed through
  3 rounds of message passing over S1 nodes, record nodes and (S1, source) groups:
  leave-one-out mean, max, and a softmax "competition" term. The output is the level-2 logit
  plus a correction whose last layer starts at zero. Training is transductive with 4 S1 folds:
  every edge is visible, and only training-fold labels enter the loss.
- **Result.** Out of fold on the graph frame it scored 0.9828 against 0.9792 for the level-2
  stack on the same S1 (+0.0036, 95% CI [+0.0034, +0.0038]; US +0.0039, India +0.0026), with
  matching train and validation loss. On the public leaderboard it scored **0.970 against 0.976**.
- **Likely causes.** The graph frame keeps all region-less records of a country but only 15% of
  its regions, so record-side competition, the signal the GNN relies on, is not distributed as
  on test. France (23% of test pairs) has no training data, and the learned correction can
  misfire on it. The refit stack behind the GNN also lost about 11% of its training sample.
  An out-of-fold gain measured on a different frame was not a reliable guide.

---

