# Business Entity Resolution (Amazon ML Challenge)

For each Source 1 record, find the Source 2 / Source 3 records that describe the same business.
The pipeline runs on CPU (a GPU only speeds up the optional cross-encoder) and uses no external
data or services; the only download is the pretrained multilingual-e5-small weights (MIT
license, 118M parameters). The level-0 pipeline is sized for about 12M records per split on a
16 GB laptop; the stack and cross-encoder were run on a 32 GB EC2 m7i.2xlarge.

```
TSVs -> normalize (parallel, parquet cache)
     -> per country: drop noise regions, infer missing regions from the country's own records
     -> per (country, region) block: IDF-weighted sparse token index -> top-k retrieval (name / address / both)
     -> stage-1 LightGBM re-ranker (view scores + cheap name / house-number evidence) -> candidates
     -> pairwise features (rapidfuzz, sparse token overlap, rank context, target-side competition)
     -> [optional] cross-encoder logit: multilingual-e5-small (MIT) fine-tuned on raw pair text
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

## Run (from this folder)
```bash
python src/pipeline.py eda          # dataset statistics
python src/pipeline.py train        # 200k-entity train sample: blocking recall, 4-fold OOF macro F0.5,
                                    # threshold, final model -> artifacts/
python src/pipeline.py validate     # leave-one-country-out check (stands in for unseen France)
python src/pipeline.py predict      # full test set -> ../../output/*.tsv
python src/pipeline.py all          # train + predict (full reproduction of the level-0 model)
python src/pipeline.py ce-train     # cross-encoder on train sample C (GPU if present; CPU budget otherwise)
python src/pipeline.py ce-score     # cross-encoder logits for the cached train / test frames
python src/pipeline.py stack        # level-1 ensemble + level-2 meta-model, ablation, -> artifacts/stack.json
python src/pipeline.py stack-predict  # final outputs from the cached test frame with the chosen level
```
`train` and `predict` cache their model inputs (pairs, features) under `cache/frames/`, so the
four later commands never regenerate candidates. Full order: `all`, `ce-train`, `ce-score`,
`stack`, `stack-predict` (`aws/run_on_ec2.sh` all, then gpu). On a CPU-only machine the
cross-encoder trains on 300k pairs and scores only pairs with level-0 probability in
[0.02, 0.98]; `--ce-full` uses all pairs (GPU). The first `ce-train` downloads
`intfloat/multilingual-e5-small` from the Hugging Face hub (pretrained weights only).
Options: `--sample N` (train S1 sample size), `--dump-errors` (OOF FP/FN pairs to `artifacts/`),
`--baseline DIR` (paired bootstrap of per-entity OOF F0.5 against an earlier run's artifacts),
`--min-free-gb` (abort instead of swapping when free RAM falls below this, default 1.5),
`--ce-sample N` (size of the cross-encoder's training sample C, default 150k),
`--ce-band LO HI` (score only pairs with level-0 probability in [LO, HI]; widening the band later
scores only the new pairs), `--ce-full` (train on all of sample C and score every pair; GPU).

Results of the full run (EC2 m7i.2xlarge, CPU only): OOF macro F0.5 0.9840 (US 0.9864, India
0.9804) with the cross-encoder stack, 0.975 on the public leaderboard.

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
| `src/cross_encoder.py` | transformer cross-encoder (multilingual-e5-small, MIT): fine-tuning on sample C, batched scoring |
| `src/stack.py` | level-1 LightGBM / XGBoost / CatBoost on shared folds, relational features, level-2 meta-model, ablation + bootstrap selection, frame prediction |
| `src/postprocess.py` | exclusive assignment; global-threshold or per-S1 expected-F0.5 decoding |
| `src/metrics.py` | macro F0.5 as defined by the challenge (singletons included), per-entity scores, paired bootstrap, blocking recall ceiling |
| `src/mine_vocab.py` | audit trail: mines noise vocabulary (abbreviations, honorifics, legal forms) from true pairs of stage-1 sample A only |
| `src/pipeline.py` | CLI; streams one country at a time |
