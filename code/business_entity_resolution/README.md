# Business Entity Resolution (Amazon ML Challenge)

For each Source 1 record, find the Source 2 / Source 3 records that describe the same business.
The pipeline runs on CPU except the two cross-encoders, which the final run trained and scored
on a GPU. It uses no external data or services; the only downloads are the pretrained
multilingual-e5-small and multilingual-e5-base weights (MIT license, 118M and 278M parameters). The level-0 pipeline is sized for about 12M records per split on a
16 GB laptop; the final run (stack and cross-encoder included) used a 32 GB EC2 m7i.2xlarge.

```
TSVs -> normalize (parallel, parquet cache)
     -> per country: drop noise regions, infer missing regions from the country's own records
     -> per (country, region) block: IDF-weighted sparse token index -> top-k retrieval (name / address / both)
     -> stage-1 LightGBM re-ranker (view scores + cheap name / house-number evidence) -> candidates
     -> pairwise features (rapidfuzz, sparse token overlap, rank context, target-side competition)
     -> cross-encoder logits: multilingual-e5-small and -base (MIT), fine-tuned on raw pair text
     -> level 1: LightGBM + XGBoost (+ CatBoost) -> level 2: LightGBM on level-1 logits +
        relational features (the S1's other candidates: twins, other-source anchor, rank)
     -> one S1 per target record; decoder chosen on OOF: global threshold, or per-S1 expected-F0.5
     -> output/matching_results.tsv, output/candidate_pairs.tsv
```
Each stacking level is kept only if its paired bootstrap against the level below has a 95% lower
bound above zero, so `stack.json` may select plain LightGBM.

## Setup
```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data location
By default the package root (two levels above this folder) must contain:
```
student_resource/dataset/train/train_source{1,2,3}.tsv, train_ground_truth.tsv
student_resource/dataset/test/test_source{1,2,3}.tsv
```
Override with `--data-dir` / `ER_DATA_DIR`. Normalised parquet caches go to `cache/` (`--cache-dir` /
`ER_CACHE_DIR`), keyed by a hash of `normalize.py`.

## Reproduce the final submission (from this folder)
```bash
# 1. stage-1 re-ranker, trained on the standard (un-hidden) training data
python src/pipeline.py train --sample 400000                 # samples A/B/C, stage-1 re-ranker -> artifacts/stage1.txt
# 2. matcher at the test set's distractor density, reusing stage 1
python src/pipeline.py train --sample 400000 --hide-s1 auto --reuse-stage1  # level-0 model, training frames
python src/pipeline.py predict                               # test blocking + features -> cache/frames/test
# 3. cross-encoder on a GPU (we used a Colab A100): all of sample C, every train / test pair scored
python src/pipeline.py ce-train --ce-full                    # downloads e5-small once -> artifacts/ce_model
python src/pipeline.py ce-score --ce-full                    # -> cache/frames/{train,test}/ce.parquet
python src/pipeline.py ce-train --ce-full --ce-model intfloat/multilingual-e5-base --ce-name base
python src/pipeline.py ce-score --ce-full --ce-model intfloat/multilingual-e5-base --ce-name base  # + ce_base_logit
# 4. stacked matcher
python src/pipeline.py stack                                 # level 1 + level 2, ablation -> artifacts/stack.json
python src/pipeline.py stack-predict                         # -> ../../output/matching_results.tsv, candidate_pairs.tsv
```
- `--hide-s1 auto` hides a data-measured share of training S1 entities (about 18.7%) so that
  training has the test set's density of unmatched records. The hidden entities come from
  outside samples A, B and C, which is why the stage-1 model from step 1 can be reused. The run
  prints `hide-s1 auto: ... h=`.
- Step 3 needs a CUDA GPU (on an A100, e5-small takes about 25 min to train and 40 min to score,
  e5-base roughly twice as long; float16 is used
  automatically on older GPUs such as the T4). On CPU only, drop `--ce-full`: the cross-encoder
  then trains on 300k pairs and scores only pairs with level-0 probability in [0.02, 0.98]
  (the 0.977 leaderboard submission; about 5 h on 8 vCPU).
- The second cross-encoder (`--ce-name base`) is stored as its own column (`ce_base_logit`)
  and model dir (`artifacts/ce_model_base`); every column of `ce.parquet` is a stack feature.
- `train` and `predict` cache their model inputs under `cache/frames/`, so the later
  commands never regenerate candidates.
- A guard aborts a command when free RAM falls below `--min-free-gb` (default 1.5); pass
  `--min-free-gb 0` on a dedicated machine. The full run needs about 30 GB of RAM (EC2
  m7i.2xlarge).

Other commands and options:
```bash
python src/pipeline.py eda          # dataset statistics
python src/pipeline.py validate     # leave-one-country-out check (stands in for unseen France)
python src/pipeline.py all          # train + predict (level-0 model only)
```
`--sample N` (size of the matcher's training sample B, default 200k), `--ce-sample N` (size of
the cross-encoder's sample C, default 150k), `--ce-band LO HI` (widening the band later scores
only the new pairs), `--reuse-stage1` (reuse `artifacts/stage1.txt`), `--use-cat` (add CatBoost
to level 1 in `stack`), `--baseline DIR` (paired bootstrap of per-entity OOF F0.5 against an
earlier run), `--dump-errors` (OOF false positives / negatives to `artifacts/`).

Results of the final run (EC2 m7i.2xlarge for everything but step 3): OOF macro F0.5 0.9858
(US 0.9878, India 0.9827) at test density, 0.979 on the public leaderboard.

Then run the official checker from `student_resource/`:
```bash
python utils/validate_submission.py --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```
Build the final zip with `python src/package_submission.py --team <team_name>`.

## Code map
| file | role |
|---|---|
| `src/config.py` | paths, blocking sizes, sample size, LightGBM params |
| `src/io_utils.py` | TSV I/O (`sep="\t"`, strings only), ground truth as a link frame, output writing and checks |
| `src/normalize.py` | romanisation of native scripts, legal forms, domains, abbreviations, landmarks, numbers, states |
| `src/blocking.py` | per-country token index, sparse top-k retrieval, target-side best-S1 scores |
| `src/features.py` | vectorised pair features |
| `src/model.py` | GroupKFold LightGBM, refit, save/load |
| `src/regions.py` | per-country region sanity (rare regions dropped) and region inference learned from the split's own records -- no place-name tables, same code for every country |
| `src/cross_encoder.py` | transformer cross-encoders (multilingual-e5-small / -base, MIT): fine-tuning on sample C, batched scoring |
| `src/stack.py` | level-1 LightGBM / XGBoost / CatBoost on shared folds, relational features, level-2 meta-model, ablation + bootstrap selection, frame prediction |
| `src/postprocess.py` | exclusive assignment; global-threshold or per-S1 expected-F0.5 decoding |
| `src/metrics.py` | macro F0.5 as defined by the challenge (singletons included), per-entity scores, paired bootstrap, blocking recall ceiling |
| `src/mine_vocab.py` | audit trail: mines noise vocabulary (abbreviations, honorifics, legal forms) from true pairs of stage-1 sample A only |
| `src/pipeline.py` | CLI; streams one country at a time |
