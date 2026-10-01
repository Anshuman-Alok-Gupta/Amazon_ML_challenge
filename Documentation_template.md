# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Ouch Ouch  
**Team Members:** Pranjal Jayesh Bhamare, Anshuman Gupta, Rishi Gandhi, Anushka Dipak Pandore  
**Submission Date:** 27/09/2026

---

## 1. Executive Summary
Our pipeline has three stages: candidate generation (blocking), a stacked pairwise matcher, and
an assignment step tuned for the precision-heavy metric.

- **Normalisation.** Country-agnostic rules romanise native-script names, standardise legal
  forms and address abbreviations, and remove junk such as PO boxes, placeholders and domain
  suffixes.
- **Blocking.** For each country and region, an IDF-weighted sparse token index retrieves a
  deep pool of Source 2/3 records under three views: name, address, and both combined.
- **Stage-1 re-ranker.** A LightGBM model re-scores that pool from the retrieval scores and
  cheap name and house-number evidence, and keeps the best 8 candidates per (S1, source).
- **Matcher.** A two-level stack:
  - Level 1: LightGBM and XGBoost on 72 similarity and competition features, plus the scores
    of two fine-tuned multilingual transformer cross-encoders that read the raw text of both
    records: `intfloat/multilingual-e5-small` (MIT, 118M parameters) and
    `intfloat/multilingual-e5-base` (MIT, 278M parameters). Both are trained on a GPU and score
    every candidate pair.
  - Level 2: a LightGBM meta-model that also looks at each S1's other candidates.
- **Assignment.** Each Source 2/3 record is given to at most one Source 1 entity, and a decoder
  tuned directly for macro F0.5 decides how many candidates each S1 keeps.

**Training at test density.** The test set has about 24% more unmatched records per S1 than the
training data. We therefore hide 18.7% of the training S1 entities, a fraction measured from
the row counts, so the model trains against the same density of distractors (Section 4).

**Results.**

| | Out-of-fold macro F0.5 | Public leaderboard |
|---|---|---|
| Stack with CPU cross-encoder, standard training data | 0.9840 (US 0.9864, India 0.9804) | 0.975 |
| Same stack at test distractor density | 0.9838 (US 0.9862, India 0.9804) | 0.977 |
| **Two cross-encoders trained on GPU, every pair scored (final)** | **0.9858 (US 0.9878, India 0.9827)** | **0.979** |

The first OOF score is measured on a different training frame. The density-matched frame is
harder, so its OOF is slightly lower even though the model scores higher on the leaderboard.
Candidate-pair recall is 98.26%. Everything runs on CPU (EC2 m7i.2xlarge: 8 vCPU, 30 GB RAM)
except the cross-encoders, which were trained and scored on a single GPU (Colab A100).

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory analysis of the training split (2.21M S1, 5.03M S2 and 5.29M S3 records):

- **Link structure.** There are 7.64M true links: 3.46 matches per S1 on average (at most 11),
  and 5.6% of S1 entities have none. No S2/S3 record is linked to more than one S1 entity, so
  the assignment can be exclusive. About 26% of S2/S3 records are unmatched distractors.
- **Country.** Every true pair shares its country label, so blocking by country is safe. We
  still treat country as an open set of labels, since France appears only in the test set.
- **Name noise.** Legal suffixes are dropped, abbreviated or moved to the front ("Pvt. EFS
  Print Ventures Ltd."). Words are reordered, misspelled, or given stray accents ("Léarning").
  Junk prefixes (`--`, `<<`) and website forms ("oncologyphysicians.com") appear. Some trade
  names share no token with the legal name ("Excellent & Co" ↔ "Orbiaria"). About 23% of Indian
  S2/S3 names are written in native scripts (Devanagari, Tamil, Telugu, Kannada, Gujarati,
  Bengali).
- **Address noise.** Components are reordered, upper-cased or abbreviated. "Street" sometimes
  becomes "Saint", and house numbers are truncated (1631 → 163). Cities are misspelled, and
  `PO BOX`, `<NULL>` and `##` appear. State names can be in native script, and landmarks are
  used as addresses ("Near Sayaji Hotel"). About 3.3% of S2/S3 addresses are empty, and US S1
  addresses carry no ZIP code.
- **Token overlap.** 99.8% of true pairs share a name token or at least two address tokens.
  This is why we chose token-based retrieval.
- **Train/test shift.** The test set has about 24% more S2/S3 records per S1 than the training
  data: India 5.82 vs 4.68, US 5.76 vs 4.67, and France 5.53. We predict about 3.4 matches per
  S1 on test, the same as the training rate. So the extra test records are unmatched
  distractors: about 42% of test S2/S3 records against 26% in training. This explains most of
  the gap between our out-of-fold and leaderboard scores (Section 5), and it is addressed by
  training at test density (Section 4).

### 2.2 Solution Strategy
**Approach Type:** blocking + learned re-ranking + stacked matcher (gradient-boosted trees and
two transformer cross-encoders) + constrained assignment

**Core Innovations:**
- Fast IDF-weighted sparse retrieval with three views, per country and region.
- "Target-side competition" features, computed against the *full* S1 pool. A model trained on a
  400k-entity sample then sees the same competition for each record as it will on the full
  test set.
- Two cross-encoders that compare the two records token by token, across scripts.
- Relational features that let the meta-model judge an S1's candidates together: the
  near-identical "twin" distractor, and the best match in the other source.
- Training at the test set's distractor density, with the hidden fraction measured from the
  data rather than hand-set.
- Decision rules tuned directly on out-of-fold macro F0.5, singletons included.
- A component is kept only if a paired bootstrap shows it helps.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** each record becomes a set of tagged tokens:
  - name tokens;
  - a compact name with the words joined, which catches domain-style names;
  - address tokens (words and numbers);
  - a consonant skeleton of the whole compact name (a `k|` token), so that transliterations
    meet: "sevnkeyr" (romanised "सेवन केयर") and "sevencare" share one. The compact-name token
    is also emitted for one-word names such as "glistensons.com".

  The tokens are IDF-weighted and L2-normalised into three views: name, address, and
  name+address. Tokens seen only once are dropped. Tokens in more than 20,000 records get zero
  weight for retrieval. For each S1 record and each target source, the top-k records under
  every view are retrieved with a multithreaded sparse top-n matrix product (`sparse_dot_topn`,
  Apache-2.0). Blocking runs per country and then per region (state). Records whose region is
  missing or unrecognised go to a residual block.
- **Region handling.** Region detection uses the same rules for every country, and none of them
  names a particular state. A region is detected only from an explicit region name or code,
  under three structural rules:
  - A segment counts as a region only if its letters form a region name and its digits, if
    any, look like a postal code. So "Fl 0" (a floor) and ".../Hr/16" (a house number) are not
    read as Florida or Haryana.
  - A trailing short word becomes a region code only in the address's last segment or next to a
    postal code. So "Oak Ct" becomes "oak court", not Connecticut.
  - If region-only segments disagree ("Oor, …, West Bengal"; "Washington, DC" vs "DC,
    Washington"), all candidates are kept. Each country then picks the one most common among
    its own unambiguous S1 records, so both orderings land in the same block.

  A region holding under 0.05% of its country's S1 records is treated as noise and sent to the
  residual block. These rules reduced the share of true training pairs split across two region
  blocks, which can never be matched, from 0.089% to 0.031% in India and from 0.349% to 0.063%
  in the US.
- **Inferring missing regions (`regions.py`).** Some sources give only a city or a sub-region
  (a département, a county). Such a record would fall into the residual block and compete with
  every S1 in the country.
  - For each country, the pipeline learns from its own records which end-of-address words
    almost always come with one region: at least 20 records and at least 98% purity.
  - It then assigns that region to region-less records whose clues all agree, in two passes. No
    labels, place-name lists or external data are used.
  - When we hid known regions and let it infer them, it was right 99.65% of the time in India
    and 99.43% in the US.
  - On the test set it cut region-less French S2/S3 records from **66.7% to 6.2%**.
- **Vocabulary fixes.**
  - Doubled letters are collapsed before dictionary lookup. Before this fix, every hand-written
    key containing a double letter could never match, among them `poona→pune`, `puducherry`,
    `null` and `estt`. Keys and values are now collapsed the same way automatically.
  - We added honorifics (`shri`, `sri`, `smt`, `mr`, `dr`, a leading "M/s") and romanised
    native-script legal forms (`pra`, `praibhet`, `limird`, and `li` after another legal form).
  - Each one was mined from true pairs of the stage-1 sample only (`mine_vocab.py`, support ≥ 50),
    never from the entities the OOF score is measured on.
- **Stage-1 re-ranking (learned).** The retrieved pool (about 116 records per S1) is scored by a
  LightGBM model with 300 rounds, trained on a 60k-S1 sample kept apart from the matcher's
  training sample.
  - Besides the three cosine scores and their ranks within the pool, it uses cheap evidence on
    each pair:
    - name token-set similarity (rapidfuzz);
    - IDF coverage of the name on both sides, and whole-name skeleton overlap;
    - shared house numbers (count and Jaccard), and whether the first numbers are equal;
    - whether the target name is in a native script, and whether its address is empty.
  - This separates a same-address record with a garbled or native-script name, or one whose
    house number changed, from the many same-name decoys in a region.
  - It keeps the top 8 per (S1, source) with probability ≥ 0.005. This set is exactly what the
    matcher scores, and it is what `candidate_pairs.tsv` contains.
  - Compared with our first re-ranker (cosine features only, top 6), it raised candidate-pair
    recall from 97.86% to **98.25%** and cut the candidates by **39%** (12.4 → 7.7 per S1).
- **Candidate pairs generated:**
  - Test: **15,814,996 pairs for 1,732,544 S1 entities (9.1 per S1)**: France 259,452 S1,
    India 809,986 and US 663,106. Only 41 S1 entities have no candidate.
  - Training sample (400k S1): 3,061,330 pairs (7.65 per S1, 44.4% positive) on the standard
    data, and 3,089,527 pairs (7.72 per S1) at test density.
  - Reduction ratio against the full cross product within each country: above 0.99999.
- **How you ensured true matches were not lost:** the three views complement each other. The
  name view keeps a match whose address is missing, and the address view keeps one with a trade
  name or a native-script name. Recall on held-out training entities:
  - retrieval pool recall: **98.92%** (sample A, 60k S1);
  - candidate-pair recall after re-ranking: **98.25%** (sample B, 400k S1), and 98.26% at test
    density;
  - this recall limits the reachable macro F0.5 to **0.9944**.

---

## 4. Matching Model

**Level-0 features (72, all country-agnostic and vectorised):**
- **Name:** rapidfuzz ratio, partial ratio, token-sort, token-set and Jaro-Winkler on the core
  name (legal forms removed). Also: compact-name ratio (for domain names); IDF-weighted token
  Jaccard, coverage, largest shared IDF and unshared IDF mass; agreement and conflict of legal
  forms; first-token match; token counts and length ratio; a native-script flag; and how common
  the name is in each pool (for chains).
- **Address:** fuzzy ratios on the cleaned address and its letters-only part; IDF-weighted
  address-token overlap; shared and conflicting numbers; first-number equality and prefix
  match (for truncated house numbers); postal-code equality; landmark similarity; and a
  missing-address flag.
- **Transliteration and aliases:** ratio and Jaro-Winkler on the consonant skeletons of the
  full compact names; IDF overlap of the whole-name skeleton token; and the best token-set score
  when either side gives an alternate name ("formerly known as", "DBA", "trading as").
- **Context:** retrieval cosines (name, address, combined) and the stage-1 score; the pair's
  rank and gap to the best among the S1's candidates, per source and overall; the number of
  candidates. For the target record: its best and second-best combined score against *all* S1
  records, and this pair's gap to that best (target-side competition).

**Cross-encoder (`src/cross_encoder.py`):**
- `intfloat/multilingual-e5-small` (MIT, 118M parameters), fine-tuned to output one score for a
  pair, from the raw text `"<name> | <address>"` of both records (at most 128 tokens).
- It sees native scripts, French and house numbers directly, which the hand-made similarity
  scores can only approximate.
- **Training:**
  - Data: all 1.16M candidate pairs of a third training sample C (150k S1; 2% held out). C is
    disjoint from the stage-1 and matcher samples, so its scores on the matcher's training data
    are out-of-sample.
  - Settings: 1 epoch, AdamW (learning rate 5e-5 with linear warm-up), batch size 64, mixed
    precision (bfloat16 on the A100; float16 with loss scaling on older GPUs).
  - The 250k-token word-embedding matrix is frozen, to save memory and keep the pretrained
    multilingual vocabulary intact.
  - On held-out pairs: **AUC 0.9986**, log-loss 0.046, accuracy 98.3%.
- **Scoring:** every candidate pair, 3.09M training and 15.8M test pairs, at about 8,300 pairs
  per second on the A100.
- **Second cross-encoder:** `intfloat/multilingual-e5-base` (MIT, 278M parameters), trained and
  applied exactly the same way (held-out AUC 0.9988, log-loss 0.043, accuracy 98.4%; about 4,100
  pairs per second). Its score is a separate stack feature, `ce_base_logit`.
- **Why two cross-encoders:** models of different sizes make partly different mistakes, so the
  stack can combine their opinions. Like every other component, the second model had to pass
  the paired bootstrap: it improved out-of-fold macro F0.5 by +0.0006 (95% CI [+0.0005,
  +0.0007]), in both countries. Together the two cross-encoders add +0.0071 over the tree model
  alone.
- **Earlier CPU version:** before we had a GPU, the cross-encoder was trained on 300k pairs
  (AUC 0.9972) and scored only pairs with level-0 probability between 0.02 and 0.98 (14% of
  test pairs); the rest were left missing. That band skipped exactly the confident
  near-duplicate distractors. Training on all of sample C on a GPU and scoring every pair
  raised the leaderboard score from 0.977 to 0.979.

**Stack (`src/stack.py`).** All levels use the same 4 folds, grouped by S1 entity (GroupKFold).
- **Level 1:** LightGBM (MIT) and XGBoost (Apache-2.0) on the level-0 features plus the two
  cross-encoder scores.
- **Level 2:** LightGBM on the level-0 features, the level-1 scores, and 21 relational
  features. These describe the level-1 probabilities of the *same S1's other* candidates:
  - rank, gap to the best, and number of likely candidates, within the S1 and within the source;
  - **twin:** the best other candidate from the same source: its probability, its name and
    address similarity, and whether its house number matches the S1. A near-identical sibling
    with a higher probability is the typical false merge.
  - **anchor:** the S1's best candidate from the *other* source, with its probability and its
    similarity to this candidate. A true S2 match usually resembles the true S3 match, which
    rescues garbled names and empty addresses.

  We left out relational features across a record's competing S1 entities. The training frame
  holds only a sample of S1, so those features would look different on the full test set.
- A third relational round was tried and rejected by the bootstrap.

**Training at the test set's distractor density (`train --hide-s1 auto`):**
- The whole pipeline runs with a fraction h of the training S1 entities hidden: blocking,
  target-side features, stage 1 and the matcher. Their S2/S3 records become unmatched
  distractors, as on test.
- h is measured from the row counts, the same way for every country:
  h = 1 − (train targets per S1) / (test targets per S1) = 1 − 4.677 / 5.754 = 0.187, which
  hides 413,344 of 2,206,821 training S1 entities.
- The hidden entities are drawn from outside samples A, B and C. So the stage-1 model is reused
  unchanged, the cross-encoders' training sample C stays disjoint from the matcher's sample, and
  the ground truth of the matcher's sample is unaffected.

**Threshold selection method:** each target record is first kept only for its
highest-probability S1. Two decoders are then compared on out-of-fold predictions, and the one
with the higher macro F0.5 is used:
- a global probability cut-off;
- per-S1 expected-F0.5 decoding. For each S1 it keeps the top k candidates that maximise
  1.25·Σpᵢ(top k) / (0.25·Σpᵢ(all) + k), and it predicts an empty list when ∏(1 − pᵢ) is higher.

Each decoder has exactly one tuned number, found by grid search on macro F0.5 (singletons
included). We take the middle of the best plateau, which is more robust than the single best
point. The final model uses expected-F0.5 decoding with a floor of 0.55.

**Guarding against overfitting:**
- Every change and every stacking level is accepted only by a paired bootstrap of per-entity
  OOF F0.5 against the previous model. The lower 95% bound of the gain must be above zero, and
  no country's mean gain may be negative.
- No country-specific thresholds, models or features are used.
- The stage-1 re-ranker, the cross-encoders and the matcher are trained on three disjoint S1
  samples. Vocabulary is mined only from entities outside the scored sample.
- The public leaderboard is not used for tuning.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), out of fold:** 4-fold GroupKFold over a 400k-S1 training sample, with
  exclusive assignment. What each component added:

  | Model | Standard data, CPU cross-encoder | Gain (95% CI) | Test density, CPU cross-encoder | Test density, GPU cross-encoders (final) | Gain (95% CI) |
  |---|---|---|---|---|---|
  | LightGBM with cosine-only re-ranker (earlier round) | 0.9779 | – | – | – | – |
  | + learned stage-1 re-ranker | 0.9792 | +0.0012 [+0.0011, +0.0014] | 0.9782 | 0.9782 | – |
  | + cross-encoder score(s) (level-1 LightGBM) | 0.9832 | +0.0040 [+0.0038, +0.0042] | 0.9829 | 0.9853 | +0.0071 [+0.0069, +0.0073] |
  | + XGBoost (level-1 mean) | 0.9833 | +0.0001 [+0.0000, +0.0001] | 0.9830 | 0.9853 | +0.0000 [+0.0000, +0.0001] |
  | + level-2 relational meta-model | 0.9840 | +0.0008 [+0.0007, +0.0009] | 0.9838 | **0.9858** | +0.0005 [+0.0004, +0.0006] |

  The final model by country:

  | Scope | S1 entities | OOF macro F0.5 |
  |---|---|---|
  | **Overall** | 400,000 | **0.9858** |
  | US | 239,845 | 0.9878 |
  | India | 160,155 | 0.9827 |

  Without the cross-encoder, the stack scores 0.9815 on the standard data (US 0.9847, India
  0.9768).
- **Leaderboard vs out-of-fold:**

  | Submission | OOF | Public leaderboard |
  |---|---|---|
  | First round | 0.9739 | 0.958 |
  | Normalisation round | 0.9779 | 0.964 |
  | Stack without cross-encoder | 0.9815 | 0.967 |
  | Stack with cross-encoder | 0.9840 | 0.975 |
  | Same stack, trained at test density | 0.9836 | 0.976 |
  | Same, cross-encoder band widened to 0.02–0.98 | 0.9838 | 0.977 |
  | **Two cross-encoders trained on GPU, every pair scored (final)** | **0.9858** | **0.979** |

  - While both scores rose, the gap between OOF and the leaderboard stayed roughly constant at
    1.4–1.6 points, until the cross-encoder narrowed it to 0.9. A constant gap points to a
    systematic train/test difference rather than overfitting.
  - The difference we found is the distractor density described in Section 2.1.
  - The model's own uncertainty rises with it. The share of S1 entities with a candidate at
    probability 0.2–0.9 is 30% on India test against 19% in training, 19% on US test against
    16%, and 32% on France.
  - The cross-encoder, which compares the raw text of both records, copes best with the extra
    near-duplicates. Its improvements gained at least as much on the leaderboard as out of fold:
    +0.008 vs +0.0025 when it was added, and +0.002 vs +0.0020 when the cross-encoders were
    trained on a GPU and applied to every pair.
  - Training at test density then gained on the leaderboard even though its OOF, now measured
    on a harder frame, went down slightly.
- **Unseen country (France):** there are no labels, but France's match rate and number of
  matches per entity are close to those of the labelled countries:

  | Test country | S1 entities | S1 with ≥ 1 match | Avg matches per S1 |
  |---|---|---|---|
  | France | 259,452 | 94.3% | 3.27 |
  | India | 809,986 | 94.0% | 3.30 |
  | US | 663,106 | 94.2% | 3.37 |

  Overall, 1,630,426 of the 1,732,544 test S1 entities (94.1%) have at least one match, with
  5,753,236 matched pairs in total. In the training data, 94.4% of S1 entities have at least one
  true link. With the GPU cross-encoders, France's matches per S1 fell the most (3.36 → 3.27),
  consistent with more French near-duplicates being rejected.
- **Error analysis** (level-0 model, OOF errors for 20,000 S1 entities). About 93% of wrong
  pairs are missed matches, which is the intended trade-off under F0.5.
  - **Common false positives (wrong merges)** are near-duplicate distractors: the same or a
    misspelled name at almost the same address, with one house or flat number changed ("1003
    Anmoltower" ↔ "1005 Anmoltower", "3300 Commonwealth Drive" ↔ "3301 …", "171 Ram Vihar" ↔ "17
    Ram Vihar"). A dropped legal form is another pattern ("Black Technology Private Limited" ↔
    "Black Technology Limited"). The twin features and the cross-encoder target exactly these
    cases.
  - **Common false negatives (missed matches):**
    - *Lost in blocking:* target names in native script ("బ్రైట్ ఇన్ఫోటెక్ ప్రైవేట్ లిమిటెడ్" ↔
      "Bright Infotech Private Limited"), and random or garbled target names where only the
      address matches ("QUOSOLVEO, 3615 Diane Ln" ↔ "Gomez Pioneer Co, 3617 Diane Lane"). The
      learned stage-1 re-ranker recovered part of these (recall 97.86% → 98.25%).
    - *Scored below the cut-off:* more than half of these have an empty or placeholder target
      address. Without an address, the model cannot rule out a same-name distractor, so it
      stays cautious. Other cases are changed legal forms, IDs or phone numbers appended to the
      name, and renumbered streets.

---

## 6. Conclusion
A learned blocking stage and a stacked matcher reach an out-of-fold macro F0.5 of 0.9840 and a
public leaderboard score of 0.975. Training at the test set's distractor density raised the
leaderboard score to 0.976, scoring more pairs with the cross-encoder raised it to 0.977, and
training two cross-encoders on a GPU on all of sample C and scoring every pair raised it to
0.979 (OOF 0.9858). The only external artefacts are two MIT-licensed pretrained encoders;
everything except the cross-encoders runs on CPU.

The largest single gain came from the transformer cross-encoder (+0.004 OOF, +0.008 on the
leaderboard). Other gains came from the stage-1 re-ranker's cheap name and house-number
evidence (+0.0012 OOF), from the relational features that judge each candidate against the S1's
other candidates (+0.0008 OOF), and from matching the test set's distractor density (+0.001 on
the leaderboard).

The remaining losses, in order of size:
1. A remaining gap between out-of-fold and leaderboard scores. Distractors remain the main
   source of wrong merges, and France has no labels to learn its patterns from.
2. Records with a missing address that are scored below the cut-off.
3. Native-script and trade-name pairs lost in blocking; the candidate set limits F0.5 to
   0.9944.
4. Cross-encoder capacity: the second, larger model added +0.0006 OOF, so the e5 family is
   flattening out. A much stronger pair model (for example an instruction-tuned LLM of up to 8B
   parameters re-scoring the uncertain pairs) is the most promising next step.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: all source code is in `src/`, and the entry point is
`src/pipeline.py`.
- `train` and `predict`: blocking, features and the level-0 model. They cache their model
  inputs under `cache/frames/`, so the later commands never regenerate candidates.
- `ce-train` and `ce-score` (with `--ce-full`, on a GPU): the two cross-encoders.
- `stack` and `stack-predict`: the stacked matcher. `stack-predict` writes
  `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- `README.md` gives the exact commands for the final submission, and `requirements.txt` the
  pinned versions.
- Learned models: LightGBM (MIT), XGBoost (Apache-2.0), and the fine-tuned
  `intfloat/multilingual-e5-small` (MIT, 118M parameters) and `intfloat/multilingual-e5-base`
  (MIT, 278M parameters); pretrained weights from the Hugging Face hub. No external data, APIs or
  lookups are used.

### B. Additional Results
Timings of the full run on AWS EC2 `m7i.2xlarge` (8 vCPU, 30 GB RAM, no GPU). Some steps ran
at the same time. The final, density-matched run repeated `train`, `predict` and the stack on
the same machine; its two cross-encoders were trained and scored on a Colab A100 (last
cross-encoder rows).

| Stage | Detail | Time |
|---|---|---|
| Normalise train | 2.21M S1, 5.03M S2, 5.29M S3 rows | 235 s |
| Stage-1 pools + fit | 60k S1 sample A; pool recall 98.92% | 811 s |
| Candidates + features | samples B (400k) and C (150k) | 1,239 s |
| Level-0 CV + refit | 3.06M pairs, 72 features, 4 folds | 657 s |
| Test blocking + features | 15.81M candidate pairs (sharing the CPU with cross-encoder training) | 3,082 s |
| Cross-encoder training (CPU version) | 300k pairs, 1 epoch, 4 threads | 140 min |
| Cross-encoder scoring (CPU version) | 362k train + 2.17M test pairs (two passes) | about 140 min |
| Cross-encoder e5-small, final (Colab A100) | train on 1.14M pairs; score 3.09M train + 15.8M test pairs | 25 + 40 min |
| Cross-encoder e5-base, final (Colab A100, in parallel) | same data | about 20 + 80 min |
| Stack (levels 1–3, ablation) | 3.06M pairs | 34 min |
| Stack prediction | 15.81M test pairs | about 12 min |

Top level-0 features by gain: `stage1`, `addr_tset`, `t_gap_best`, `comb_cos`, `t_margin`,
`legal_conflict`, `name_idf_cov_t`, `legal_equal`, `name_idf_unshared`, `first_num_prefix`,
`t_is_best`, `first_num_equal`.

The official validator (`utils/validate_submission.py`, run with `--check-ids`) reports
**PASS** on the final submission. All 1,732,544 S1 entities appear in both files:
`matching_results.tsv` has 1,630,426 non-empty rows, and `candidate_pairs.tsv` has 1,732,503.

### C. Rejected Experiment: Graph Neural Network
The candidate pairs form a bipartite graph between S1 entities and S2/S3 records. A graph
neural network (GNN) can use what the other candidates of the same S1, and the other S1
entities claiming the same record, say about a pair. We tried one, and **did not use it**
because it scored lower on the leaderboard. The code is on the repository's `gnn` branch.

- **Training graph.** The stack's training frame holds a sample of about 18% of S1 entities,
  so most of a record's competing S1 entities are missing from it. The GNN was therefore trained
  on a separate frame: every S1 entity of a seeded 15% of the training regions (193k S1, 1.54M
  pairs), with the same 18.7% of S1 hidden as in the final model. The stack was refit without
  those S1 entities, so its scores on this frame are out-of-sample.
- **Model.** Each pair (edge) starts from the level-2 inputs and score, plus node degrees. Three
  rounds of message passing then run over S1 nodes, record nodes and (S1, source) groups, using
  a leave-one-out mean, a max, and a softmax "competition" term. The output is the level-2 score
  plus a learned correction whose last layer starts at zero. Training is transductive with 4 S1
  folds: every edge is visible, and only the training folds' labels enter the loss.
- **Result.** Out of fold on this frame it scored 0.9828 against 0.9792 for the level-2 stack
  on the same S1 entities (+0.0036, 95% CI [+0.0034, +0.0038]; US +0.0039, India +0.0026), with
  matching training and validation loss. On the public leaderboard it scored **0.970**, against
  0.976 for the same stack without it.
- **Likely causes.**
  - The GNN's frame keeps all region-less records of a country but only 15% of its regions.
    So competition for a record, the signal the GNN relies on, is not distributed as on test.
  - France, 23% of test pairs, has no training data, and the learned correction can misfire
    on it.
  - The refit stack behind the GNN also lost about 11% of its training sample.
  - The lesson: an out-of-fold gain measured on a different frame was not a reliable guide.
