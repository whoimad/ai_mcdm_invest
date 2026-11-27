"""Go/no-go gate: the checks the going-concern extraction must pass on training
years before any fiscal 2017 or 2018 filing is read (decision 6.9).

All on fiscal 2014–2016 10-Ks, with the training run's results as input:

- Canaries: paragraphs with a known answer planted in clear auditor's
  reports, a quarter in the largest companies' 10-Ks, so a flag shows the
  model follows the text and not its memory. At least 95% must be right.
- Quotes: every quote the gate's reads return is found in its chunk.
- Stability: 100 10-Ks read three times, cache bypassed, give identical labels.
- Locate miss rate: 200 10-Ks read again with Items 8 and 15 in full; the
  share flagged only then is measured and reported, with no pass bar.

Run with `uv run python -m ai_mcdm_invest.llm.gate` after the training run.
It needs Ollama with `qwen2.5:7b` and a GPU. If a check fails, the prompt or
locate rule is fixed on training years and the gate is run again.
"""

import re
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.corpus import iter_record_sections
from ai_mcdm_invest.llm.going_concern import (
    CLEAR,
    FAILED,
    FLAGGED,
    LLMClient,
    extract_going_concern,
    filing_status,
    quote_found,
)
from ai_mcdm_invest.llm.locate import (
    REPORT_ITEMS,
    Chunk,
    locate_auditor_reports,
    read_in_full,
)
from ai_mcdm_invest.llm.ollama import OllamaClient
from ai_mcdm_invest.llm.training_run import (
    KEY,
    TRAINING_YEARS,
    SectionsLoader,
    located_chunks,
)
from ai_mcdm_invest.tables import write_table

SAMPLE_SEED = 0
RUNS = 3
# Pass bar: the share of canaries the extraction must get right.
CANARY_BAR = 0.95
# Hosts are drawn with spares, since a report without an opinion sentence
# takes no canary.
SPARE_HOSTS = 2

# The four kinds of canary and the answer each must get. Only a genuine
# auditor's doubt about this company for this year is flagged; the other three
# are the known traps (decision 6.9).
EXPECTED = {"genuine": FLAGGED, "boilerplate": CLEAR, "stale": CLEAR, "other_company": CLEAR}

# Several wordings per kind, modelled on real auditor's reports (PCAOB AS 2415)
# and on the accounting rule's text (ASU 2014-15). `{year_end}` is the fiscal
# year end the 10-K covers, `{prior_year_end}` the one before, and
# `{prior_report_date}` the date of the auditor's report on that earlier year.
CANARY_WORDINGS = {
    "genuine": [
        "The accompanying financial statements have been prepared assuming that the Company "
        "will continue as a going concern. As discussed in Note 2 to the financial statements, "
        "the Company has suffered recurring losses from operations and has a net capital "
        "deficiency that raise substantial doubt about its ability to continue as a going "
        "concern. Management's plans in regard to these matters are also described in Note 2. "
        "The financial statements do not include any adjustments that might result from the "
        "outcome of this uncertainty.",
        "The accompanying consolidated financial statements have been prepared assuming that "
        "the Company will continue as a going concern. As discussed in Note 1, the Company is "
        "in default under its credit agreement as of {year_end} and does not have sufficient "
        "liquidity to repay the amounts due. These conditions raise substantial doubt about "
        "the Company's ability to continue as a going concern. The consolidated financial "
        "statements do not include any adjustments that might result from the outcome of "
        "this uncertainty.",
        "The accompanying financial statements have been prepared assuming that the Company "
        "will continue as a going concern. As of {year_end}, the Company has negative working "
        "capital and an accumulated deficit, and it has incurred net losses and negative cash "
        "flows from operations for the year then ended. These matters raise substantial doubt "
        "about the Company's ability to continue as a going concern. Management's plans "
        "concerning these matters are described in Note 3.",
    ],
    "boilerplate": [
        "In August 2014, the FASB issued ASU 2014-15, Presentation of Financial Statements - "
        "Going Concern, which requires management to evaluate whether there are conditions or "
        "events that raise substantial doubt about an entity's ability to continue as a going "
        "concern within one year after the date that the financial statements are issued. The "
        "Company does not expect the adoption of this guidance to have a material effect on "
        "its financial statements.",
        "Management has evaluated whether there are conditions or events, considered in the "
        "aggregate, that raise substantial doubt about the Company's ability to continue as a "
        "going concern within one year after the date that the financial statements are "
        "issued. Management concluded that no such conditions or events exist as of "
        "{year_end}.",
        "The Company incurred a net loss for the year ended {year_end}. Management's plans, "
        "which include a new credit facility closed after year end, alleviate the substantial "
        "doubt about the Company's ability to continue as a going concern, and management has "
        "concluded that no substantial doubt remains.",
    ],
    "stale": [
        "In our report dated {prior_report_date}, we expressed an opinion that the financial "
        "statements for the year ended {prior_year_end} were prepared assuming that the "
        "Company would continue as a going concern, and that its recurring losses from "
        "operations raised substantial doubt about its ability to continue as a going concern. "
        "As discussed in Note 2, the Company has since completed an equity offering. "
        "Accordingly, our present opinion on those financial statements, as presented herein, "
        "is different from that expressed in our previous report.",
        "The financial statements for the year ended {prior_year_end} were audited by other "
        "auditors, whose report dated {prior_report_date} included an explanatory paragraph "
        "stating that the Company's recurring losses from operations raised substantial doubt "
        "about its ability to continue as a going concern at that date.",
        "In our report dated {prior_report_date}, we stated that the Company's recurring "
        "losses and defaults under its loan agreements raised substantial doubt about its "
        "ability to continue as a going concern. As discussed in Note 3, the Company has "
        "since refinanced its debt, and the conditions that raised substantial doubt about "
        "the Company's ability to continue as a going concern at {prior_year_end} no longer "
        "exist.",
    ],
    "other_company": [
        "We did not audit the financial statements of Northwind Holdings, Inc., an equity "
        "method investee of the Company. Those statements were audited by other auditors, "
        "whose report included an explanatory paragraph stating that Northwind Holdings, "
        "Inc. has suffered recurring losses from operations that raise substantial doubt "
        "about Northwind Holdings, Inc.'s ability to continue as a going concern.",
        "As discussed in Note 7, the Company's largest customer, Bellwether Retail Corp., "
        "has disclosed that its recurring losses and defaults under its loan agreements raise "
        "substantial doubt about Bellwether Retail Corp.'s ability to continue as a going "
        "concern. The Company's receivable from Bellwether Retail Corp. is described in "
        "Note 7.",
        "As discussed in Note 4, the Company sold its interest in Calder Mining Ltd. during "
        "the year ended {year_end}. The financial statements of Calder Mining Ltd. state that "
        "its recurring losses raise substantial doubt about Calder Mining Ltd.'s ability to "
        "continue as a going concern.",
    ],
}
# The auditor signs about ten weeks after the fiscal year end.
REPORT_LAG = timedelta(days=75)

# The end of the auditor's opinion sentence: "present fairly", then the next
# period that ends a sentence.
_OPINION_END = re.compile(r"\bpresents?,?\s+fairly.*?\.(?=\s)", re.IGNORECASE | re.DOTALL)


def canary_paragraph(kind: str, variant: int, period_end: str) -> str:
    """Wording `variant` of a canary of `kind`, for a 10-K covering the fiscal
    year ended `period_end` (ISO date)."""
    year_end = date.fromisoformat(period_end)
    try:
        prior_year_end = year_end.replace(year=year_end.year - 1)
    except ValueError:  # 29 February becomes 28 February the year before
        prior_year_end = year_end.replace(year=year_end.year - 1, day=28)
    return CANARY_WORDINGS[kind][variant].format(
        year_end=_long_date(year_end),
        prior_year_end=_long_date(prior_year_end),
        prior_report_date=_long_date(prior_year_end + REPORT_LAG),
    )


def _long_date(day: date) -> str:
    return f"{day:%B} {day.day}, {day.year}"


def plant_canary(sections: dict[str, str], paragraph: str) -> dict[str, str] | None:
    """`sections` (Item -> text) with `paragraph` planted in the first located
    auditor's report, just after its opinion sentence, so the model reads it.

    None when there is nowhere to plant it: no located report (a fallback
    read), no opinion sentence in the report, or a paragraph that would change
    where the report ends.
    """
    located = locate_auditor_reports(sections)
    if located.fallback_read:
        return None
    chunk = located.chunks[0]
    opinion = _OPINION_END.search(chunk.text)
    if opinion is None:
        return None
    text = sections[chunk.item]
    at = text.index(chunk.text) + opinion.end()
    insert = f"\n{paragraph}\n"
    planted = sections | {chunk.item: text[:at] + insert + text[at:]}
    expected = chunk.text[: opinion.end()] + insert + chunk.text[opinion.end() :]
    if expected not in [found.text for found in locate_auditor_reports(planted).chunks]:
        return None
    return planted


@dataclass(frozen=True)
class Gate:
    """The gate's outcome: one row per check (`check`, `value`, `bar`,
    `passed`), and the rows each check was measured on."""

    checks: pd.DataFrame
    canaries: pd.DataFrame
    stability: pd.DataFrame
    full_reads: pd.DataFrame


def run_gate(
    training: pd.DataFrame,
    filings: pd.DataFrame,
    market_caps: pd.DataFrame,
    load_sections: SectionsLoader,
    client: LLMClient,
    store_dir: Path,
    *,
    canaries: int = 200,
    well_known: int = 40,
    stable: int = 100,
    full_reads: int = 200,
) -> Gate:
    """Run every gate check on the training years.

    `training` is the training-run filings table (see `training_run`),
    `filings` the filings table (for `period_end` and `corpus_year`) and
    `market_caps` holds `market_cap` per `cik` and `fiscal_year`.

    Canaries: `canaries` clear 10-Ks, `well_known` of them the largest
    companies, each get one planted paragraph with a known answer.

    Stability: `stable` 10-Ks, half of them flagged in the training run, are
    read `RUNS` times, each time with an empty cache.

    Locate miss rate: `full_reads` random 10-Ks with a located report and a
    training-run label are read again with Items 8 and 15 in full. A miss is
    one flagged only by the full read.
    """
    table = training.merge(
        filings.dropna(subset=["accession"])[["accession", "period_end", "corpus_year"]],
        on="accession",
        validate="1:1",
    ).merge(market_caps[[*KEY, "market_cap"]], on=KEY, how="left", validate="1:1")
    groups = _host_candidates(table, canaries, well_known)
    stable_sample = _stability_sample(table, stable)
    full_sample = _full_read_sample(table, full_reads)
    # One pass over the corpus for every 10-K the gate reads.
    sections = {
        key: {item: texts.get(item, "") for item in REPORT_ITEMS}
        for key, texts in load_sections(
            _corpus_keys(pd.concat([*groups.values(), stable_sample, full_sample]))
        )
    }

    planted = _plant(groups, sections, canaries, well_known)
    quotes: list[bool] = []  # whether each quote the gate reads is in its chunk
    canary_results = _read_canaries(planted, client, store_dir / "canaries", quotes)
    stability = _stability(stable_sample, sections, client, quotes)
    full = _read_in_full(full_sample, sections, client, store_dir / "full_reads", quotes)

    correct = int(canary_results["correct"].sum())
    share = correct / len(canary_results)
    identical = int(stability["identical"].sum())
    missed = int(full["missed"].sum())
    found = sum(quotes)
    checks = pd.DataFrame(
        [
            {
                "check": "canaries",
                "value": f"{correct} of {len(canary_results)} correct ({share:.1%})",
                "bar": f"at least {CANARY_BAR:.0%} correct",
                "passed": share >= CANARY_BAR,
            },
            {
                "check": "quotes",
                "value": f"{found} of {len(quotes)} quotes found in their chunk",
                "bar": "every quote found in its chunk",
                "passed": found == len(quotes),
            },
            {
                "check": "stability",
                "value": f"{identical} of {len(stability)} 10-Ks identical in {RUNS} runs",
                "bar": f"every 10-K identical in {RUNS} runs, cache bypassed",
                "passed": identical == len(stability),
            },
            {
                "check": "locate_miss_rate",
                "value": f"{missed} of {len(full)} 10-Ks ({missed / len(full):.1%}) "
                "flagged only when read in full",
                "bar": "measured and reported",
                "passed": None,
            },
        ]
    )
    return Gate(checks=checks, canaries=canary_results, stability=stability, full_reads=full)


def _host_candidates(
    table: pd.DataFrame, canaries: int, well_known: int
) -> dict[bool, pd.DataFrame]:
    """Clear 10-Ks with a located report that the keyword benchmark does not
    flag either, one per company: the largest companies first (`True`), then
    the others in a fixed random order (`False`), with spares."""
    hosts = table[(table["status"] == CLEAR) & ~table["fallback_read"] & ~table["keyword_flagged"]]
    largest = (
        hosts.dropna(subset=["market_cap"])
        .sort_values("market_cap", ascending=False)
        .drop_duplicates("cik")
        .head(well_known * SPARE_HOSTS)
    )
    # A spare large company left unused may still be drawn at random.
    others = (
        hosts[~hosts["cik"].isin(largest["cik"].head(well_known))]
        .sample(frac=1, random_state=SAMPLE_SEED)
        .drop_duplicates("cik")
        .head((canaries - well_known) * SPARE_HOSTS)
    )
    return {True: largest, False: others}


def _corpus_keys(rows: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(rows["cik"].astype(int), rows["corpus_year"].astype(int), strict=True))


def _plant(
    groups: dict[bool, pd.DataFrame],
    sections: dict[tuple[int, int], dict[str, str]],
    canaries: int,
    well_known: int,
) -> list[dict]:
    """One canary per host, kinds and wordings taken in turn within each group."""
    kinds = list(CANARY_WORDINGS)
    planted = []
    used = set()
    for is_well_known, wanted in [(True, well_known), (False, canaries - well_known)]:
        count = 0
        for host in groups[is_well_known].to_dict("records"):
            if count == wanted:
                break
            if host["cik"] in used:
                continue
            kind = kinds[count % len(kinds)]
            variant = (count // len(kinds)) % len(CANARY_WORDINGS[kind])
            period_end = _period_end(host)
            paragraph = canary_paragraph(kind, variant, period_end)
            host_sections = sections.get((int(host["cik"]), int(host["corpus_year"])), {})
            with_canary = plant_canary(
                {item: host_sections.get(item, "") for item in REPORT_ITEMS}, paragraph
            )
            if with_canary is None:
                continue
            planted.append(
                {key: host[key] for key in [*KEY, "accession"]}
                | {
                    "well_known": is_well_known,
                    "kind": kind,
                    "variant": variant,
                    "expected": EXPECTED[kind],
                    "period_end": period_end,
                    "sections": with_canary,
                }
            )
            used.add(host["cik"])
            count += 1
    return planted


def _chunk_table(filings: list[tuple[str, str, tuple[Chunk, ...]]]) -> pd.DataFrame:
    """The extraction's input from (`filing_id`, `period_end`, chunks) triples."""
    return pd.DataFrame(
        [
            {
                "filing_id": filing_id,
                "period_end": period_end,
                "chunk": number,
                "item": chunk.item,
                "text": chunk.text,
            }
            for filing_id, period_end, chunks in filings
            for number, chunk in enumerate(chunks)
        ],
        columns=["filing_id", "period_end", "chunk", "item", "text"],
    )


def _extract(
    chunks: pd.DataFrame, client: LLMClient, store_dir: Path, quotes: list[bool]
) -> pd.DataFrame:
    """The extraction's results, each quote checked again against the chunk it
    cites and the outcome added to `quotes`."""
    results = extract_going_concern(chunks, client, store_dir)
    cited = results.merge(chunks[["filing_id", "chunk", "text"]], on=["filing_id", "chunk"])
    quotes += [
        quote_found(quote, text)
        for quote, text in zip(cited["quote"], cited["text"], strict=True)
        if isinstance(quote, str) and quote
    ]
    return results


def _period_end(row: dict) -> str:
    return pd.Timestamp(row["period_end"]).date().isoformat()


def _read_canaries(
    planted: list[dict], client: LLMClient, store_dir: Path, quotes: list[bool]
) -> pd.DataFrame:
    """Each canary's status from the extraction, and whether it is the expected one."""
    chunks = _chunk_table(
        [
            (canary["accession"], canary["period_end"], located.chunks)
            for canary in planted
            for located in [locate_auditor_reports(canary["sections"])]
        ]
    )
    results = _extract(chunks, client, store_dir, quotes)
    status = filing_status(results)
    # The canary is planted in the first chunk; its quote is that chunk's.
    quote = results[results["chunk"] == 0].set_index("filing_id")["quote"]
    table = pd.DataFrame(planted).drop(columns=["sections", "period_end"])
    return table.assign(
        status=table["accession"].map(status),
        correct=lambda rows: rows["status"] == rows["expected"],
        quote=table["accession"].map(quote),
    )


def _stability_sample(table: pd.DataFrame, stable: int) -> pd.DataFrame:
    """Half flagged in the training run, the rest read and not flagged, so the
    runs are not mostly "clear" against "clear"."""
    flagged = table[table["status"] == FLAGGED]
    flagged = flagged.sample(n=min(stable // 2, len(flagged)), random_state=SAMPLE_SEED)
    others = table[table["status"].isin([CLEAR, FAILED])]
    others = others.sample(n=min(stable - len(flagged), len(others)), random_state=SAMPLE_SEED)
    return pd.concat([flagged, others], ignore_index=True)


def _stability(
    sample: pd.DataFrame,
    sections: dict[tuple[int, int], dict[str, str]],
    client: LLMClient,
    quotes: list[bool],
) -> pd.DataFrame:
    """Each sampled 10-K's status in each of `RUNS` runs, every run with an
    empty cache, so every run asks the model again."""
    chunks, _ = located_chunks(sample, _serve(sections))
    table = sample[[*KEY, "accession"]].assign(training_status=sample["status"])
    for run in range(1, RUNS + 1):
        with tempfile.TemporaryDirectory() as empty_store:
            status = filing_status(_extract(chunks, client, Path(empty_store), quotes))
        table[f"run_{run}"] = table["accession"].map(status)
    runs = table[[f"run_{run}" for run in range(1, RUNS + 1)]]
    return table.assign(identical=runs.eq(runs["run_1"], axis=0).all(axis=1))


def _serve(sections: dict[tuple[int, int], dict[str, str]]) -> SectionsLoader:
    """A loader over 10-K text already in memory."""
    return lambda keys: ((key, sections[key]) for key in keys if key in sections)


def _full_read_sample(table: pd.DataFrame, full_reads: int) -> pd.DataFrame:
    """Random 10-Ks whose report was located and labelled flagged or clear."""
    located = table[~table["fallback_read"] & table["status"].isin([FLAGGED, CLEAR])]
    return located.sample(n=min(full_reads, len(located)), random_state=SAMPLE_SEED)


def _read_in_full(
    sample: pd.DataFrame,
    sections: dict[tuple[int, int], dict[str, str]],
    client: LLMClient,
    store_dir: Path,
    quotes: list[bool],
) -> pd.DataFrame:
    """Each sampled 10-K's status when Items 8 and 15 are read in full, beside
    its located status from the training run."""
    chunks = _chunk_table(
        [
            (
                row["accession"],
                _period_end(row),
                read_in_full(sections.get((int(row["cik"]), int(row["corpus_year"])), {})),
            )
            for row in sample.to_dict("records")
        ]
    )
    results = _extract(chunks, client, store_dir, quotes)
    status = filing_status(results)
    flagged = results[results["status"] == FLAGGED].drop_duplicates("filing_id")
    table = sample[[*KEY, "accession"]].assign(
        located_status=sample["status"],
        full_status=sample["accession"].map(status),
        full_quote=sample["accession"].map(flagged.set_index("filing_id")["quote"]),
    )
    return table.assign(
        missed=(table["full_status"] == FLAGGED) & (table["located_status"] != FLAGGED)
    ).reset_index(drop=True)


def gate_report(gate: Gate, model_version: str) -> str:
    """The gate report: the verdict, pass or fail per check, and the canaries
    by kind and by host."""
    passed = gate.checks["passed"].dropna().astype(bool).all()
    verdict = {True: "PASS", False: "FAIL", None: "reported"}
    checks = [
        f"| {row['check']} | {row['value']} | {row['bar']} | {verdict[row['passed']]} |"
        for row in gate.checks.to_dict("records")
    ]
    canaries = gate.canaries
    by_kind = [_accuracy_row(kind, canaries[canaries["kind"] == kind]) for kind in CANARY_WORDINGS]
    by_host = [
        _accuracy_row(name, canaries[canaries["well_known"] == is_well_known])
        for name, is_well_known in [("well-known companies", True), ("other companies", False)]
    ]
    next_step = (
        "The test years may be read."
        if passed
        else "Fix the prompt or the locate rule on training years, then run the gate again."
    )
    return "\n".join(
        [
            f"# Going-concern gate, fiscal {TRAINING_YEARS[0]}-{TRAINING_YEARS[-1]}",
            "",
            f"Model: `{model_version}`",
            "",
            f"**Gate: {verdict[bool(passed)]}**. {next_step}",
            "",
            "| Check | Result | Pass bar | Outcome |",
            "|---|---|---|---|",
            *checks,
            "",
            "Canaries by kind (genuine doubt must be flagged, the three traps clear):",
            "",
            "| Kind | Canaries | Correct | Share |",
            "|---|---|---|---|",
            *by_kind,
            "",
            "Canaries by host:",
            "",
            "| Host | Canaries | Correct | Share |",
            "|---|---|---|---|",
            *by_host,
            "",
        ]
    )


def _accuracy_row(name: str, canaries: pd.DataFrame) -> str:
    correct = int(canaries["correct"].sum())
    share = correct / len(canaries) if len(canaries) else float("nan")
    return f"| {name} | {len(canaries)} | {correct} | {share:.1%} |"


def main() -> None:
    settings = Settings.from_env()
    client = OllamaClient()
    print(f"model: {client.model_version}")
    panel = pd.read_parquet(settings.processed_dir / "panel.parquet")
    gate = run_gate(
        pd.read_parquet(settings.processed_dir / "going_concern_training_filings.parquet"),
        pd.read_parquet(settings.processed_dir / "filings.parquet"),
        panel[[*KEY, "Market Cap"]].rename(columns={"Market Cap": "market_cap"}),
        lambda keys: iter_record_sections(settings.raw_corpus_dir, keys),
        client,
        settings.interim_dir / "going_concern_gate_store",
    )
    for name in ["checks", "canaries", "stability", "full_reads"]:
        write_table(getattr(gate, name), settings.processed_dir, f"going_concern_gate_{name}")
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report = settings.reports_dir / "going_concern_gate.md"
    report.write_text(gate_report(gate, client.model_version))
    print(report.read_text())


if __name__ == "__main__":
    main()
