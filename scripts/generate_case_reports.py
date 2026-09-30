from pathlib import Path
from html import escape

import polars as pl


INPUT_PATH = Path(
    "data/processed/migration_assessments.parquet"
)

OUTPUT_DIR = Path(
    "reports/migrations"
)


def value(value):
    if value is None:
        return "N/A"
    return str(value)


def build_report(row: dict) -> str:

    repo = escape(value(row.get("repo_name")))
    old_version = escape(value(row.get("old_version")))
    new_version = escape(value(row.get("new_version")))

    segment = escape(
        value(row.get("difficulty_segment"))
    )

    indicator_count = value(
        row.get("observed_indicator_count")
    )

    assessment = escape(
        value(row.get("assessment"))
    )

    evidence = escape(
        value(row.get("evidence"))
    )

    old_commit = escape(
        value(row.get("old_commit"))
    )

    new_commit = escape(
        value(row.get("new_commit"))
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MIGRO Migration Assessment</title>

<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 900px;
    margin: 50px auto;
    padding: 0 24px;
    color: #202124;
    line-height: 1.6;
}}

h1 {{
    margin-bottom: 4px;
}}

.subtitle {{
    color: #666;
    margin-bottom: 32px;
}}

.card {{
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 20px;
    margin: 18px 0;
}}

.label {{
    font-size: 12px;
    text-transform: uppercase;
    color: #777;
    letter-spacing: 0.08em;
}}

.value {{
    font-size: 20px;
    font-weight: 600;
}}

.evidence {{
    background: #f6f6f6;
    padding: 16px;
    border-radius: 8px;
}}

.note {{
    color: #666;
    font-size: 14px;
}}
</style>
</head>

<body>

<h1>MIGRO Migration Assessment</h1>

<div class="subtitle">
Evidence-based software migration intelligence
</div>

<div class="card">
    <div class="label">Repository</div>
    <div class="value">{repo}</div>
</div>

<div class="card">
    <div class="label">Migration</div>
    <div class="value">
        Java {old_version} → Java {new_version}
    </div>
</div>

<div class="card">
    <div class="label">Observed difficulty segment</div>
    <div class="value">{segment}</div>
</div>

<div class="card">
    <div class="label">Elevated indicators</div>
    <div class="value">{indicator_count}</div>
</div>

<div class="card">
    <div class="label">Assessment</div>
    <div class="value">{assessment}</div>
</div>

<div class="card">
    <div class="label">Evidence</div>

    <div class="evidence">
        {evidence}
    </div>
</div>

<div class="card">
    <div class="label">Migration commits</div>

    <p>
        <strong>Before:</strong><br>
        {old_commit}
    </p>

    <p>
        <strong>After:</strong><br>
        {new_commit}
    </p>
</div>

<div class="card">
    <div class="label">Interpretation</div>

    <p>
        This assessment summarizes observable characteristics
        of the migration in relation to the empirical
        distributions present in the analyzed dataset.
    </p>

    <p class="note">
        The assessment is descriptive. It does not establish
        causality, probability of migration failure, or future
        project outcomes.
    </p>
</div>

<div class="card">
    <div class="label">Limitations</div>

    <p class="note">
        MIGRO operates on the available migration records and
        patch information. Repository history, engineering
        effort, developer experience, issue discussions, CI
        failures, and undocumented migration work may not be
        represented.
    </p>
</div>

</body>
</html>
"""


def main() -> None:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    df = pl.read_parquet(INPUT_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("MIGRO - CASE REPORT GENERATOR")
    print("=" * 70)

    print(f"\nMigrations: {df.height:,}")

    generated = 0

    for index, row in enumerate(
        df.iter_rows(named=True),
        start=1,
    ):

        repo = row.get("repo_name") or "unknown_repo"

        safe_repo = "".join(
            character
            if character.isalnum()
            or character in "-_."
            else "_"
            for character in str(repo)
        )

        filename = (
            f"{index:04d}_{safe_repo}.html"
        )

        path = OUTPUT_DIR / filename

        path.write_text(
            build_report(row),
            encoding="utf-8",
        )

        generated += 1

    print("\n" + "-" * 70)
    print("REPORTS")
    print("-" * 70)

    print(f"Generated: {generated:,}")
    print(f"Directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
