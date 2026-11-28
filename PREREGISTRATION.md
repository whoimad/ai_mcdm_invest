# Pre-registration: the test-year run (ticket 16, issue #6)

## 1. Hypotheses

- **H1.** The LLM going-concern flag adds information on next-year tail loss beyond the ML risk score.
- **H2.** The keyword benchmark's flag (`kw-1`) adds information beyond the ML risk score.
- **H3.** The LLM flag adds information beyond both the ML risk score and the `kw-1` flag.

A tail loss is a next-year return in the worst 20% of its fiscal year.

## 2. Evaluation universe and test years

Test years: fiscal 2017 and 2018, pooled. The universe is joined-panel rows (linked or recovered companies) whose 10-K text is in the corpus, whose Items 8 and 15 hold audit opinion text, and that report at least half the ratios. Every layer, baseline and test uses the same rows.

| Step (2017 + 2018) | Rows | Tail-loss rate |
|---|---|---|
| Joined panel | 8,240 | 20.5% |
| Excluded: no 10-K text in the corpus | 1,480 | 17.0% |
| Excluded: "no report" (no audit opinion text) | 172 | 12.2% |
| Excluded: too few ratios | 58 | 10.3% |
| **Evaluation universe** (3,259 + 3,271) | **6,530** | **21.6%** |
| of which fallback reads | 71 | 14.1% |

Filings the LLM records as "extraction failed" stay in the universe. They are listed with their tail-loss rate and left out of the tests in section 4; in the ranking they score on their other criteria.

## 3. Weight profiles and rescaling rule

Fixed before testing, by criterion group, never fitted to returns (`mcdm/ranking.py`).

| Profile | Profitability | Valuation | Balance sheet | ML risk | Text |
|---|---|---|---|---|---|
| Balanced | 20% | 20% | 20% | 20% | 20% |
| Risk-averse | 10% | 10% | 20% | 40% | 20% |
| Value-minded | 25% | 35% | 10% | 20% | 10% |

A group's weight is split equally over its criteria in use. A group with no criterion in use (its layer switched off, or every criterion dropped) loses its weight, and the rest are rescaled to sum to 100%. Example, risk-averse, ratios only: 25 / 25 / 50. The text group holds one criterion, the LLM flag (clear scores 1, flagged 0). Rankings use SAW.

## 4. Primary test

For each hypothesis, a likelihood-ratio test (1 degree of freedom) of one added flag in a logistic model of tail loss, pooled over 2017–2018, at α = 0.05:

- **Base model:** tail loss ~ logit(ML tail-loss probability) + a 2018 dummy. The probability is the walk-forward XGBoost score (trained on 2015–2016 for 2017, on 2015–2017 for 2018).
- **H1:** base + LLM flag. **H2:** base + `kw-1` flag. **H3:** base + `kw-1` flag + LLM flag, tested against base + `kw-1` flag.

Each test reports the LR statistic, its p-value, and the flag's odds ratio with a 95% interval. No correction for three tests; all three are reported.

## 5. Fixed pipeline

- Prompt `gc-2`; check `check-2`; locate rule of `llm/locate.py` (heading within Items 8 and 15, opinion within 1,500 characters, chunks of about 6k tokens, fallback read in full when no heading).
- Benchmark `kw-1`, frozen, run on the same located chunks.
- Model `qwen2.5:7b@sha256:845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, temperature 0, seed 42, 32,768-token context, schema-constrained JSON.
- Scoring rule: a chunk is flagged only when it is the auditor's opinion, about this company, about the current fiscal year, and doubt is expressed. `alleviated` is stored and does not change the score. A 10-K is flagged when any chunk is, else failed when any chunk failed, else clear.

**What was not redone.** The training-year run and `kw-1`'s tuning used prompt `gc-1`. The gc-1 gate failed on stale opinions (19 of 50 correct). `gc-2` changed only the prompt wording for stale and alleviated doubt, and asks `alleviated` before `doubt_expressed`. The scoring rule did not change. The training years were not rerun with `gc-2`.

## 6. Gate result

**PASS** on `gc-2` (run of 2026-11-27, `data/reports/going_concern_gate.md`): canaries 200 of 200 (genuine, boilerplate, stale and other-company 50 of 50 each; well-known hosts 40 of 40), quotes 402 of 402 found in their chunk, 100 of 100 10-Ks identical in 3 runs with the cache bypassed, locate miss rate 0 of 200. The `gc-1` gate of 2026-11-26 failed: canaries 165 of 200 (stale 19 of 50, boilerplate 46 of 50).

Decided before the `gc-2` gate result was known: the test years would be run once even if `gc-2` failed the 95% canary bar, and the result reported as is.

## 7. Descriptive only

Reported without a test, and labelled descriptive:

- AUC gain of "+ LLM" over "+ ML" (point value, no bootstrap).
- The ablation table (ratios only → + ML → + LLM, three profiles, beside the size and last-year-move baselines).
- The table of filings where the LLM and `kw-1` disagree, with quotes.
- The share of LLM-flagged companies already in ML's riskiest fifth.
- Robustness: the ablation with TOPSIS, and without fallback reads.

The ML layer's numbers stay marked exploratory. Nulls are reported: "the LLM reads correctly but adds little beyond ML" is a valid finding.

## 8. Cut and not attempted

- Risk-factor trial (ticket 15).
- Referee model on disagreements (ticket 19).
- Weight-sensitivity figure, and the ablation without non-December year ends (ticket 20).

## 9. Review

Drafted by Claude on 2026-11-27 and pushed before the test-year run; the owner reviews it after the push. Any change after the run is logged below with its reason.

### Changes after the run

-
