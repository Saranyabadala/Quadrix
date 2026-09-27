# Supervised Entity Resolution across three noisy business sources

For every Source 1 entity, find its matching records in Source 2 and Source 3.
Zero, one, or many matches per source. No shared key, so matching is done on
learned similarity across the fields that actually exist.

Two pipelines live in this repo:

| mode | schema | data | entry point |
|---|---|---|---|
| `--mode er` (default) | `entity_id` / `business_name` / `business_address` / `country` | `student_resource/dataset`, 2.2M / 5.0M / 5.3M rows | `python run_pipeline.py --mode er` |
| `--mode benchmark` | the richer synthetic schema (phone, category, geo, …) | `data/raw`, generated | `python run_pipeline.py --mode benchmark --generate` |

**F_0.5 is the only reported evaluation metric in both.** No precision, recall,
F1, AUC, accuracy or log-loss number is printed or returned anywhere in the
reporting layer.

## The real-schema pipeline (`--mode er`)

```bash
# train, tune the threshold, and report one F_0.5 on held-out entities
python run_pipeline.py --mode er --er-config config/er.yaml

# score the unlabeled test set with the model the run above saved
python run_pipeline.py --mode er --er-mode predict --model-path artifacts/er/model

# a fast, coherent slice of the real training data (positives + hard negatives)
python tools/build_er_subsample.py --n-s1 15000 --n-neg 120000 --out artifacts/er_subsample
python run_pipeline.py --mode er --er-config artifacts/er_subsample/er_subsample.yaml \
    --out-dir artifacts/er_subsample/out --work-dir artifacts/er_subsample/work
```

Everything is configured in `config/er.yaml`: paths, column mapping, chunk size,
every blocking rule and cap, the composite weights, both thresholds, and the
model parameters. Nothing about this pipeline is hard-coded.

### The one-paragraph version

Index Source 1 once → stream Source 2 and Source 3 in chunks and probe each chunk
against **four** Source 1 indexes, unioned (**OR**, not AND) → compute 28
pairwise features per candidate from the four real columns → train a
gradient-boosted binary classifier on labelled pairs → tune the decision
threshold as the **F_0.5 maximizer** over 0.10…0.90 → run a looser second pass
over the records that got **zero** candidates, so no record is ever dropped →
score every candidate and turn the winner into one of four tiers (`matched` /
`manual_review` / `ambiguous` / `unmatched`) → write per-record outputs, a
per-entity submission table, a candidate debug log, and a summary report
containing exactly one metric: **F_0.5**.

On a 15,000-entity / 172k-record coherent subsample of the real training data:
**F_0.5 = 0.894** on a held-out set of Source 1 entities that influenced neither
the model nor the threshold, with blocking recall **0.886** on the same data.

---

## Multi-index blocking and the OR logic

No single key over four columns has both recall and a small block, so
`build_source1_indexes` builds four and `generate_candidate_pairs` takes their
**union**. A Source 2/3 record becomes a candidate for a Source 1 entity if
**any** rule fires:

| # | rule | key | why it is there |
|---|---|---|---|
| 1 | `country_prefix` | `country \| first 4 chars of business_name` | the workhorse: same country, same brand prefix |
| 2 | `country_city_prefix` | `country \| extracted city \| name prefix` | most precise rule — re-anchors on the locality extracted from the address blob |
| 3 | `shared_token` | `country \| token`, for every significant name token | survives abbreviations ("Acme Bicycle" / "Acme Bike"); the highest-recall rule |
| 4 | `phonetic_country` | `country \| nysiis(business_name)` | survives spelling errors ("Hendersen" / "Henderson") |
| 5 | `exact_entity_id` | `entity_id` | configurable hook, **off by default**: the three sources come from independent registries, so it is not expected to fire |

**Why OR and not AND.** An AND keeps only the pairs that agree on *every* key —
which is exactly the set that needs no ensemble. Each rule fails somewhere
different, so their union is what carries the recall, and the classifier's job is
to separate the useful pairs from the noise inside each block.

Every rule is country-scoped, and the token index is keyed `country|token` by
default, so a "Paris" in France cannot meet a "Paris" in Texas. Set
`scope_token_index_by_country: false` for a bare global token index — more recall
when a country label is wrong, at the cost of large cross-country blocks.

**No empty value is ever a key.** A rule contributes only when every part of its
key is present, and a bucket held by more than `max_bucket_size` Source 1 rows is
dropped and counted. A key built from a blank country would be a key shared by
every record that lacks one — a cross product wearing a disguise.

### Candidate-size safety (`MAX_CANDIDATES_PER_RECORD`)

`max_candidates_per_record` (default 100) is a ceiling, not a filter. When a
record exceeds it:

1. Any **rule** above `per_rule_cap` (default 40) is narrowed with an extra cheap
   key — the candidate must also share a significant name token — and re-merged.
   The rule is never dropped: dropping a rule loses a real record, narrowing it
   only loses the unidentifiable tail.
2. If the merged set is still over the cap it is cut in rule-precision order
   (city+prefix first, token last) and **logged**, in `processing_notes` and as a
   `truncated_to_cap` row in `candidate_debug_log.csv`. A truncation is a
   decision a reviewer has to be able to see.
3. Rows above `large_candidate_multiplier × cap` are logged as
   `large_candidate_set` whether or not they were truncated.

`match_count_per_blocking_rule` in the summary report then shows which rules
earned their cost: a rule with many candidate pairs and no matches is pure
overhead, and a rule with few pairs carrying most of the matches is the one to
keep while the others tighten.

### Fallback: why no record is ever dropped

`fallback_candidate_generation` runs **only** on records the first pass left at
zero candidates, and only ever *adds* candidates. Two country-scoped mechanisms,
both `O(log n)` per record:

- `fallback_fuzzy_name` — binary-search the record's normalized name into its
  country's sorted name list, take ±`fallback_window` neighbours, keep those
  whose `token_set_ratio` clears `fallback_fuzzy_floor` (0.72).
- `fallback_sorted_neighborhood` — the immediate ±`fallback_neighborhood_window`
  neighbours with no similarity floor at all, for names mangled past the fuzzy bar
  that still sort next to their twin.

**These thresholds generate candidates; they never decide a match.** Every pair
the fallback produces must still clear the same F_0.5-tuned classifier threshold
as any other pair, and it is flagged `blk_fallback = 1` so the model can learn to
discount it.

Records still at zero after the fallback go to `unresolved_records.csv` **and**
appear in `unmatched_records.csv` with `generation_status =
unresolved_after_fallback` in `processing_notes`. They are never dropped: a lost
record is a silent, permanent error; a listed one is a work item.

### Tuning the blocking rules from manually reviewed matches

`summary_report.json` gives you the two numbers to watch:

- `validation.blocking_recall` — the fraction of true pairs the candidate stage
  produced at all. This is a hard ceiling on recall: a pair blocking never
  generated cannot be recovered by any model. On the subsample it is **0.886**,
  so 11% of true matches are currently unreachable.
- `match_count_per_blocking_rule` — candidates vs accepted matches per rule.

Read them together: a low F_0.5 with high blocking recall is a *model* problem
(adjust features); a low F_0.5 with low blocking recall is a *Stage 1* problem
(adjust rules). Then, from the review files:

1. Look at `unresolved_records.csv` first. If those rows have names that *look*
   related to something in Source 1, a rule is missing — usually a shorter
   `name_prefix_len` or a higher `max_token_df_ratio`.
2. Look at `candidate_debug_log.csv` for `truncated_to_cap` rows. Truncation
   means a coarse key flooded the record; raising `per_rule_cap` for that rule, or
   lowering `max_bucket_size`, recovers the tail.
3. Only then consider tightening: a rule contributing candidates but no matches
   can be switched off (`enable_*: false`) or given a smaller cap.

The measured trade-off on the 15k-entity subsample — this is why the shipped
defaults sit where they do:

| `max_token_df_ratio` | cutoff | blocking recall | candidates/record |
|---|---|---|---|
| 0.002 | 30 | 0.8637 | 16.1 |
| **0.01** | **150** | **0.8856** | **40.2** |
| 0.05 | 750 | 0.8759 | 50.8 |
| 0.2 | 3000 | 0.8759 | 51.0 |

Past the knee recall flattens while the candidate count keeps growing, and the
growth only pushes records into the per-record cap where truncation starts
dropping true pairs. Keep `max_token_df_absolute` at or below
`max_bucket_size`: a token admitted to the index and then dropped as a hot bucket
is pure waste.

---

## Features, and the weighted composite score

`compute_pairwise_features` builds 28 features from the four real columns, all on
0–1, or `NaN` for "no evidence" (which the trees route natively, and which the
paired `*_missing_either` flags let the model tell apart from a genuine 0):

- **name** — `token_set_ratio`, `token_sort_ratio`, Jaro-Winkler,
  `token_overlap_ratio` (containment), plus `name_similarity`, the mean of those
  four over the metrics that exist for the pair.
- **address** — `token_set_ratio` over the normalized `business_address`.
- **city / state** — similarity and an exact-match flag each.
- **country** — `country_exact`.
- **phonetic** — `phonetic_name_match`.
- **missingness** — name/address/city/state/country missing on either side,
  address-parse-failed, name-placeholder, and a total count.
- **blocking provenance** — one flag per rule, a count, and `blk_fallback`.

There is deliberately **no** phone, email, postal-code, first/last-name,
date-of-birth or gender feature: those columns do not exist in this dataset, and
a feature on a missing column is a constant that teaches the model nothing.

`compute_weighted_composite_score` is a transparent weighted average of those
features — name 0.45, address 0.30, city/state 0.15, country 0.10 — computed
alongside the classifier purely for **explainability and manual review**. When a
group's features are all missing for a pair, its weight is redistributed across
the remaining groups, so a record with no published address is not punished for
it. It is written to the output next to `classifier_probability` and **never
overrides the classifier's decision**.

### City and state, extracted from one free-text field

`business_address` is one opaque string and its layout varies between sources, so
"the city" is not a single well-defined token. The extractor returns a **set of
locality candidates** rather than one guess:

| address | candidates | primary | state |
|---|---|---|---|
| `123 N Main St, Springfield, IL 62704` | `springfield` | `springfield` | IL |
| `BROWNING, 19 VAC RD, MT` | `browning` | `browning` | MT |
| `Unit APT 1, Anchorage, AK, 1350 27th Avenue` | `anchorage`, `ak`, `1350 27th avenue` | `anchorage` | — |
| `Yadgarpally Village, Miryalguda Mandal, Survey No. 328/A/1, Miryalguda, Telangana` | `yadgarpally village`, `miryalguda mandal`, `miryalguda` | `miryalguda` | TG |
| `Near SBI ATM, MG Road, Bengaluru, Karnataka 560001` | `mg road`, `bengaluru` | `bengaluru` | KA |

`city_exact` is a **set intersection** and `city_similarity` compares the primary
guesses. That is not a detail: comparing one extracted token by equality produced
18,458 phantom "city conflict" vetoes on true matches and cost **24 F_0.5 points**
(0.654 → 0.894). A state is only read when it *trails* the address (after any
postcode), which is what stops `Draper City (sl Co), UT` from being read as state
**CO**.

---

## The tiered decision

`decide_match_status` turns one record's pair probabilities into one status:

```
classifier_probability >= high_threshold    -> matched
low_threshold <= probability < high          -> manual_review
probability < low_threshold                  -> unmatched
```

with two overrides that can demote an automatic match:

- **`ambiguous`** — the runner-up Source 1 entity is within `ambiguity_margin`
  (default 0.05) of the winner. Two Source 1 records explain the record about
  equally well, so neither is applied automatically. The alternative stays visible
  in `second_best_source1_id` / `second_best_probability`.
- **conflict veto** — the winner's country, or its city/state, *actively*
  conflicts. The status becomes `manual_review`, not `unmatched`: the model liked
  the pair, something structural disagrees, and a human is the right resolver.
  "Actively" means both sides published a value, the values differ, **and** they
  are not merely noisy spellings of each other — `location_conflict_floor` (0.80)
  keeps "High Point" / "Highpoint" out of the veto.

`high_threshold` is tuned per run. `low_threshold_ratio` (0.6) expresses the floor
as a fraction of the tuned high threshold, so the middle tier keeps its width
when the high threshold moves; a `low` above `high` is rejected at construction,
because it would make the middle tier unreachable.

### Tuning thresholds and weights with manually reviewed matches

The review files are the feedback loop:

1. Label a sample of `manual_review_records.csv`. The `ambiguous` rows are the
   most informative — two entities look equally plausible and only a person can
   say which is right.
2. Feed the labels back into the truth set used by `--er-mode validate` and
   re-run. The tuned `high_threshold` moves to the F_0.5 maximizer on the *tune*
   split; the reported F_0.5 stays on the untouched *report* split.
3. To move the high threshold by hand, set `decision.high_threshold` and run
   `--er-mode predict` — the tiering and both submission files follow it.
4. To move the floor, change `decision.low_threshold_ratio`. Raising it widens the
   review queue; lowering it trades review work for missed matches.
5. The `weights:` block affects only the explainable composite score, never the
   decision. Change it when the review files show a component being misjudged and
   you want the *explanation* to reflect that; change the *feature set* when you
   want the decision to change.

F_0.5 is precision-heavy by construction, so the usual first move when F_0.5
disappoints is to raise `high_threshold` and grow the review queue, not to
retrain.

---

## F_0.5 is the only metric

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Precision and recall are still computed — F_0.5 is derived from them and cannot be
computed without them — but they are **internal**. They are never printed on their
own and never returned as a reported result. `tune_threshold` sweeps 0.10 → 0.90
in steps of 0.05, computes F_0.5 at each point, and returns only the maximizing
threshold; `final_evaluation` prints exactly one line:

```
  F_0.5 (held-out validation) = 0.8936
```

Two details worth knowing:

- **The reported number is the per-Source-1-entity F_0.5.** The challenge metric
  is F_0.5 computed per Source 1 entity and then averaged over *all* of them,
  singletons included — a false merge on a singleton costs a full 1.0. The sweep
  inside `tune_threshold` runs on pair labels, which is the same formula at a
  different grain; the flat sweep saturates earlier than the entity metric, which
  is one reason ties are broken toward the *higher* threshold.
- **F_0.5 needs labels, so it is only computed where they exist.** It is *not*
  recomputed during final prediction on the unlabeled test set. `--er-mode
  predict` reports counts, rates and the threshold it used, and states explicitly
  in `summary_report.json` that no F_0.5 was computed.

LightGBM's internal `eval_metric` (logloss / AUC, for early stopping) is a
training mechanic, not a reported metric. It is the reporting layer that is
stripped.

## Outputs

| file | one row per | notes |
|---|---|---|
| `matched_records.csv` | auto-matched Source 2/3 record | full feature-level evidence |
| `manual_review_records.csv` | record | `manual_review` **and** `ambiguous` |
| `unmatched_records.csv` | record | decided non-matches |
| `unresolved_records.csv` | record | zero candidates even after fallback — a *different* failure with a different fix |
| `candidate_debug_log.csv` | event | oversized / truncated / still-unresolved |
| `summary_report.json` | run | per-source counts, per-status counts, match rate, avg candidates, avg probability for matches, per-rule match counts, unresolved-after-fallback, blocking recall, score distribution, and **F_0.5** |
| `matching_results.tsv` | Source 1 entity | submission: `matched_entity_ids`, empty for singletons |
| `candidate_pairs.tsv` | Source 1 entity | submission: the candidate set the model actually scored |

Every Source 1 entity appears in both submission files, including entities that
were never a candidate for anything — the metric scores singletons, so a missing
row changes the denominator rather than merely losing a match.

## Scale and cost

| stage | cost |
|---|---|
| Source 1 indexing | `defaultdict(list)` inverted indexes, one pass over 2.2M rows |
| Source 2/3 | `pandas.read_csv(chunksize=...)`, one chunk resident at a time |
| pair features | `rapidfuzz.process.cpdist` — C-level, multi-threaded, element-wise |
| model | LightGBM, or sklearn HistGB when the LightGBM wheel cannot load |

Peak memory is *the Source 1 index + one chunk + that chunk's pairs*. No
`Source1 × Source2/3` product is ever formed: the 2.2M × 10.3M cross product is
~2.3e13 pairs and is never materialized anywhere.

The run makes **two passes** over the candidate stream: pass 1 gathers a bounded
training sample and caches the tune-split rows to disk, the classifier is fitted
and the threshold tuned, then pass 2 reloads the model, decides and writes
incrementally. Holding ~30M scored pairs in memory between the two would defeat
the point of chunking.

**Known costs, measured.** The dominant per-record cost is Python-level
normalization — the address is parsed with regexes, row by row — at roughly 15µs
per row, so ~4 minutes per 10M rows per pass, and the run does two passes. A
15k-entity / 172k-record subsample run (2.2M candidate pairs per pass) takes
~4.5 minutes on 4 cores. Extrapolating to the full 10.3M-record stream gives
roughly 1.5–3 hours per pass, so plan for several hours and use
`--max-source1 / --max-source2 / --max-source3 / --max-chunks` while iterating.

---

## The benchmark pipeline (`--mode benchmark`)

## Results on the generated benchmark

1,296 Source 1 entities · 1,716 Source 2 · 1,668 Source 3 · 46,547 candidate pairs

| stage | metric | value |
|---|---|---|
| blocking | candidate pairs / cross product | 46,547 / 4,385,664 = **1.1%** |
| blocking | **recall** (true pairs surviving) | **0.964** (S2 0.972, S3 0.955) |
| features | pairs × features | 46,547 × 59 |
| model (oracle) | test **F_0.5** | **0.993** |
| **final table** | **F_0.5** vs hidden truth | **0.993** |
| output | Source 1 entities with ≥1 match | 1,238 / 1,296 (95.5%) |
| output | accepted pairs | 2,937 (6.3% of candidates) |

The precision/recall/ROC-AUC numbers this table used to carry were removed when
the reporting layer was reduced to F_0.5; they still exist as internal selection
inputs inside `evaluate.recommend_threshold` and are not printed.

**Where the remaining error lives.** Of the 3,045 true matches, blocking kept
2,935 and the model correctly matched 2,926 of those — it recovers **99.7% of
the reachable matches**, so the model is not the bottleneck. All 119 missed
matches are pairs blocking never generated. Fixing the 3.6% blocking loss is
worth more than any model tuning.

"Oracle" = trained on the hidden truth, to prove the feature set is sound. It is
not a production mode; on real data it is replaced by `--labels reviewed.csv`.


## Layout

```
config/er.yaml            every knob of the real-schema pipeline (`--mode er`)
config/schema.yaml        column mapping for the benchmark pipeline
run_pipeline.py           CLI entry point, both modes
src/normalize.py          step 1  cleaning & normalization
                           (+ real-schema: is_placeholder, locality_candidates,
                            extract_city_state, clean_entity_frame)
src/generate.py           synthetic benchmark + hidden truth
src/block.py              step 2  blocking, candidate generation, blocking recall
                           (+ multi-index blocker for the real schema:
                            build_source1_indexes, generate_candidate_pairs,
                            fallback_candidate_generation)
src/features.py           step 3  pairwise feature engineering
                           (+ real-schema: compute_pairwise_features,
                            compute_weighted_composite_score)
src/labels.py             step 4  silver labels + stratified manual labeling
src/model.py              step 5  classifier (LightGBM or sklearn HistGB), save/load
src/evaluate.py           step 6  F_0.5 threshold tuning, plots, F_0.5-only reports
src/assemble.py           step 7  tiered decision, per-record + per-entity outputs
src/metrics.py            the F_0.5 metric itself (per Source 1 entity)
src/pipeline.py           orchestration: run() for the benchmark,
                           run_er_pipeline() for the real schema
tools/build_er_subsample.py   coherent labelled subsample of the real training data
tests/                    65 tests pinning the output contract and the F_0.5-only rule
```

## Using your own data

1. Copy your three CSVs into `data/raw/`.
2. Edit `config/schema.yaml`: per source, the id column and a map from canonical
   field names to your actual column names. Use `null` for a field a source does
   not carry.
3. `python run_pipeline.py`

```yaml
source3:
  path: data/raw/source3.csv
  id_col: listing_id
  fields:
    name: listing_name
    city: null              # this source has no city column
    zip: postal_code
    lat: geo_lat            # omit lat/lon and geodistance switches itself off
```

## Design choices

### Why this blocking strategy

Six keys, unioned, because no single key has both recall and small blocks:

| key | carries | fails when |
|---|---|---|
| `phone10` | high precision | phone missing, or registry lists a switchboard |
| `zip5` | strong locality | zip missing or recorded as a PO box |
| `zip3_name` | the workhorse: relaxes zip, re-anchors on a name fragment | name changed entirely |
| `soundex` | survives small spelling errors | genuinely different wording ("Bicycle"/"Bike") |
| `name_prefix` | cheap last resort | key too coarse on common brands (see below) |
| `geo_cell` | works when the address is mangled but the record is geocoded | no coordinates (Source 2 has none) |
| `token` | same brand, different suffix | brand token too common to be informative |

Two deliberate choices:

- **Block size cap (40).** A key shared by more records than that is a sign the
  key is too coarse for the data, not that those records match. Skipped blocks
  are *counted and reported* rather than silently dropped — the manifest shows
  `name_prefix` skipped **34 of 34** blocks and `token` **74 of 74**. On this
  data those two keys are **completely inert**: with only 30 brand names every
  brand prefix is shared by ~40 entities, so neither key can ever fire. Raise
  `--max-block-size` to re-enable them and accept the quadratic blowup, or leave
  it and know they contribute nothing here. The pipeline tells you which case
  you are in instead of leaving you to guess.
- **Blocking provenance is a feature.** A pair reached by phone *and* zip *and*
  name is corroborated by independent evidence; a pair sharing one loose name
  token is not. The model gets `blk_<key>` booleans and `blk_n_keys` so it can
  tell those apart.

**Blocking recall is the gate.** 0.964 sounds fine until you notice it is
*conditional on the pairs blocking kept*: the 3.6% it dropped are
unrecoverable, and no model can find them. Missed pairs are written to
`artifacts/blocking_missed_pairs.csv` for inspection.

### Why this feature set

Three principles:

1. **Compare components, not blobs.** One fuzzy score over concatenated
   name+address hides *which part* agreed. Decomposing into house number,
   street name, suffix, unit, city, state, zip lets the model learn that a
   matching house number in the same zip is strong even when the street name
   was misspelled — and that a unit conflict is a *downgrade*, not a
   disqualifier.
2. **Several metrics, because they disagree.** Jaro-Winkler rewards a shared
   prefix, Levenshtein penalizes length change, TF-IDF cosine is corpus-aware
   (a rare brand word outweighs "shop"), token overlap ignores word order. For
   "Acme Bicycle" vs "Acme Bike" they disagree sharply, and the disagreement is
   the signal.
3. **Missingness is evidence.** A NaN in `phone10_equal` because the phone is
   missing is not the same as a NaN because the numbers differ. Every key field
   gets `_missing_either` and `_missing_both` indicators, so the model can tell
   "no evidence" from "negative evidence". Values are left as NaN and never
   imputed — the boosters route NaN natively, and imputation would invent data
   for exactly the pairs we know least about.

Two name representations are kept: `name_norm` (full cleaned name, legal suffix
retained) and `name_core` (suffixes and generic business words stripped), so
"The Acme Bicycle Co., LLC" and "Acme Bike Shop" can meet in the middle.

Oracle feature importances — 17 of 59 features carry real signal, and no single
one dominates, which is what a working feature set should look like:

| feature | share | feature | share |
|---|---|---|---|
| `phone7_equal` | 43% | `addr_lev` | 5.5% |
| `zip5_equal` | 16% | `name_jw_core` | 1.9% |
| `category_equal` | 12% | *(+ 10 more below 1.5% each)* | |
| `geo_dist_km` | 8.7% | | |
| `addr_jw` | 8.6% | | |

### Why this threshold

**Recommended rule: maximize F1 subject to precision ≥ target** (default 0.99).
Not "maximize recall subject to a precision floor" — that lands the operating
point exactly *on* the boundary, where a one-point change in the data moves it
wholesale. Restricting first and optimizing second gives an interior, stable
threshold. In practice the two candidates agreed (t=0.246, P=0.995, R=0.992).

The asymmetry is the justification. A false positive is invisible: it merges two
real businesses and corrupts everything downstream. A false negative is visible
— an unmatched row you can go find. So precision-leaning is the right default,
and `max_f1` is reported alongside for when recall matters more.

Note the threshold is **not** transferable between datasets. It landed at 0.809,
0.246 and 0.412 on three runs of the same pipeline here depending on the labels
used. Re-tune on your own validation split; never copy a threshold across
datasets or even across label sets.

Precision is reported with a **Wilson 95% interval**, because a high precision
computed from a handful of positives is misleading. At 10/10 the interval is
[0.74, 1.00] — not 1.00.

## The part that matters most: your labels are the bottleneck

There is no ground truth here, so `labels.py` produces two very different
things and keeps them clearly separate.

**Silver labels** are bootstrapped from a hand-written rule
(`labels._rule_score`). They pre-train the model and fill the unambiguous
strata. **Any metric computed on them is meaningless**, and the pipeline proves
it rather than asserting it:

| labels used | reported test precision | actual test precision |
|---|---|---|
| silver (rule-derived) | 0.998 | **0.941** |
| human-reviewed (1,200 labels) | 0.951 | 0.951 |

That 0.06 gap is not a bug — it is what training and scoring on your own
heuristic looks like. You can see the same leakage in the importances: on
silver labels the model collapses onto two features (`name_jw_core` 81%,
`blk_phone10` 18%, everything else ~1e-17) because those are precisely the
features that encode the rule. On real labels the importances spread across 17
features. If you see a two-feature importance profile, you have measured your
own heuristic, not your data.

**The real workflow** is the human-labeling sheet
(`artifacts/labeling_sheet.csv`, 1,200 pairs, `label` column blank):

```
source1_id, candidate_id, candidate_source, label, stratum, rule_score,
s1_name_raw, cand_name_raw, s1_address_raw, cand_address_raw, ...   (raw text
to decide on, plus the features behind the rule score, to audit it)
```

Fill `label` with 1/0, then:

```
python run_pipeline.py --labels artifacts/labeling_sheet.csv
```

**Sizing the budget matters.** 1,200 is not decoration — at 220 labels the
validation split is 44 rows and a reported precision of 1.00 carries a 95%
interval of roughly [0.84, 1.00], indistinguishable from 0.90. Around
1,000–1,500 labels with several hundred positives is where the interval
tightens enough to choose a threshold defensibly. If capacity is scarce, cut
`certain_match`/`certain_nonmatch` first; the accuracy comes from the hard
strata. The sheet is deliberately *not* proportional to stratum size — the
ambiguous stratum is where the decision boundary lives.

Whatever the labels say, `output/accepted_pairs_to_review.csv` gives you 200
random *accepted* pairs to eyeball. That is the only unbiased read on real
precision, and the number worth quoting once production labels exist.

## The synthetic benchmark, and why it has the traps it has

`--generate` writes three CSVs with per-source corruption profiles (registry
legal names + "Ste 200" + switchboard phone + no geo; directory marketing names
+ USPS-abbreviated streets + missing city + 7-digit phone + a different category
vocabulary) **and a hidden truth file**.

A generator that only produces easy positives flatters every downstream number,
so it includes deliberate difficulty:

- **chains** — one entity legitimately matching 2–3 records per source
- **twins** — a *different real business* with the same name and the same zip at
  a different address. The hard negative this problem is really about: city,
  state, zip, street and suffix all agree, and only the house number and phone
  separate them.
- **relocations** — same entity, *different* address, phone preserved. This is
  the deliberate contrast to a twin: name matches, address disagrees, and only
  the phone distinguishes "moved" from "different business". Without this case a
  model learns "address disagrees ⇒ non-match" and silently loses every business
  that moved.
- **address keying errors** — transposed house numbers, "123" vs "123A",
  misspelled street names
- **lexical synonyms** — "Bicycle"→"Bike", "Automotive"→"Auto", not abbreviations
- **missingness** — 2–20% per field per source

Two realism details that materially change results, both found by running the
pipeline rather than by reasoning:

- Zips are drawn from a **pool of 6 per city**. With a random zip per entity,
  `zip5_equal` is a near-perfect entity identifier: the model put 92% of its
  importance there and learned nothing about matching. Real businesses cluster
  on a few postcodes.
- `CATEGORY_SYNONYMS` in `normalize.py` is the single source of truth for
  category equivalence, and the generator derives Source 3's vocabulary *from
  it*. Two hand-maintained tables drift; this one cannot.

## Model backend

`model.py` prefers **LightGBM** and falls back to scikit-learn's
`HistGradientBoostingClassifier` — the same algorithm ported into sklearn, with
the same native NaN handling. On this machine LightGBM could not load:

```
dlopen(.../lib_lightgbm.dylib): Library not loaded: @rpath/libomp.dylib
```

The macOS wheel links LLVM's OpenMP runtime. Fix it with
`brew install libomp` (or any Linux/Windows box, where it is not an issue) and
the pipeline uses LightGBM automatically — no code change. The backend actually
used is printed on every run and recorded in the manifest, so fallback results
are never mistaken for LightGBM results.

I tried shimming `libomp` in-process. It is not worth shipping: the ABI is
version-specific, and my first attempt imported cleanly and then segfaulted
inside `__kmpc_fork_call` with a mis-read argument list. A shim that *appears*
to work is worse than none, so the honest fallback is the portable one.

## Tests

```
.venv/bin/python -m pytest tests/ -q     # 65 passed
```

Mostly contract tests, not model tests — the failure modes that would silently
corrupt results rather than raise. `tests/test_pipeline.py` pins the benchmark
output contract: zero-match Source 1 entities survive into the final table,
many-to-many matches are preserved, silver labels never invent a label in the
ambiguous middle, the Wilson interval is wide for small samples.

`tests/test_er_pipeline.py` pins the real-schema behaviour:

- city/state extraction from one free-text `business_address`, including the
  "town before street" and "state in the middle of a company name" forms, and
  placeholders never producing a key;
- OR-based candidate generation — each rule is switched on **alone** and still
  produces its pair, and rules do not cross countries;
- an over-producing rule is *narrowed*, not dropped, and the truncation reaches
  the debug log;
- the fallback fires only on zero-candidate rows, is still accountable when it
  fails, and its pairs are marked `blk_fallback`;
- ambiguous detection, the country veto, the city veto, and a spelling variant
  correctly *not* vetoed;
- `tune_threshold` returns no metric other than F_0.5, prints nothing, and
  `final_evaluation` prints exactly one line — asserted on the captured stdout,
  not just the return value.

## Known limitations

- **Address parsing is US-centric.** It handles comma-delimited, space-delimited
  and PO-box forms, and recovers multi-word cities. It does *not* peel a
  multi-word state name off the tail, because "New York"/"Virginia"/"Georgia"
  are also city names and only position disambiguates them. Guessing "state"
  destroys a high-value feature; guessing "city" costs one weak one. Where a
  dedicated city column exists it always wins. Non-US formats need a rewrite of
  `parse_address`.
- **Recall is capped by blocking**, and the cap is reported rather than
  smoothed over. 3.6% of true pairs are currently unreachable.
- **The benchmark is easier than production.** Synthetic data has clean
  structure that real registries do not. Treat 0.995/0.992 as an upper bound.

- **No cluster constraint.** A Source 1 entity can be matched to two Source 2
  records that are actually the same business, producing a duplicate that the
  dedupe step should have caught. `--margin` handles the common case; a full
  solution is correlation clustering over the accepted pairs, which is
  out of scope here.
- **Embeddings are not included.** `all-MiniLM-L6-v2` cosine on concatenated
  name+address is worth adding for the "different words, same business" cases
  (the "Bicycle"/"Bike" gap), but it needs a model download, so it is left out
  rather than made a hidden dependency.
