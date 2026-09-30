from pathlib import Path
from html import escape

import polars as pl


ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "migration_feature_matrix.parquet"
)

EVIDENCE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "migration_risk_evidence.parquet"
)

DECISIONS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "migration_decisions.parquet"
)

EVALUATION_PATH = (
    ROOT
    / "data"
    / "processed"
    / "risk_model_evaluation.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "reports"
    / "migro_executive_dashboard.html"
)


def fmt_number(value):
    if value is None:
        return "N/A"

    try:
        return f"{float(value):,.1f}"
    except Exception:
        return str(value)


def fmt_int(value):
    if value is None:
        return "N/A"

    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def pct(value):
    if value is None:
        return "N/A"

    return f"{float(value) * 100:.1f}%"


def safe_value(row, column):
    if column not in row:
        return None

    return row[column]


def card(title, value, subtitle=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-title">{escape(title)}</div>
        <div class="kpi-value">{escape(str(value))}</div>
        <div class="kpi-subtitle">{escape(subtitle)}</div>
    </div>
    """


def table(headers, rows):
    header_html = "".join(
        f"<th>{escape(str(header))}</th>"
        for header in headers
    )

    body_html = ""

    for row in rows:
        body_html += "<tr>"

        for value in row:
            body_html += (
                f"<td>{escape(str(value))}</td>"
            )

        body_html += "</tr>"

    return f"""
    <table>
        <thead>
            <tr>{header_html}</tr>
        </thead>
        <tbody>
            {body_html}
        </tbody>
    </table>
    """


def main():

    print("=" * 70)
    print("MIGRO - Executive Dashboard Generator")
    print("=" * 70)

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {FEATURES_PATH}"
        )

    if not EVIDENCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {EVIDENCE_PATH}"
        )

    if not DECISIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {DECISIONS_PATH}"
        )

    features = pl.read_parquet(
        FEATURES_PATH
    )

    evidence = pl.read_parquet(
        EVIDENCE_PATH
    )

    decisions = pl.read_parquet(
        DECISIONS_PATH
    )

    # ------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------

    total_migrations = features.height

    duration_column = (
        "migration_duration_days"
        if "migration_duration_days" in features.columns
        else "duration_days"
    )

    files_column = (
        "files_changed"
        if "files_changed" in features.columns
        else None
    )

    lines_column = (
        "changed_lines"
        if "changed_lines" in features.columns
        else None
    )

    duration_median = None
    duration_p75 = None
    files_median = None
    lines_median = None

    if duration_column in features.columns:
        duration_median = features.select(
            pl.col(duration_column).median()
        ).item()

        duration_p75 = features.select(
            pl.col(duration_column).quantile(0.75)
        ).item()

    if files_column:
        files_median = features.select(
            pl.col(files_column).median()
        ).item()

    if lines_column:
        lines_median = features.select(
            pl.col(lines_column).median()
        ).item()

    # ------------------------------------------------------------
    # High duration
    # ------------------------------------------------------------

    high_duration_rate = None

    if duration_column in features.columns:

        high_duration_rate = (
            features
            .select(
                (
                    pl.col(duration_column)
                    >= duration_p75
                )
                .mean()
            )
            .item()
        )

    # ------------------------------------------------------------
    # Evidence distribution
    # ------------------------------------------------------------

    evidence_distribution = (
        evidence
        .group_by("evidence_level")
        .len()
        .sort("len", descending=True)
    )

    evidence_rows = []

    for row in evidence_distribution.to_dicts():

        evidence_rows.append(
            [
                row["evidence_level"],
                fmt_int(row["len"]),
                pct(
                    row["len"]
                    / max(total_migrations, 1)
                ),
            ]
        )

    # ------------------------------------------------------------
    # Decision distribution
    # ------------------------------------------------------------

    decision_distribution = (
        decisions
        .group_by("decision")
        .len()
        .sort("len", descending=True)
    )

    decision_rows = []

    for row in decision_distribution.to_dicts():

        decision_rows.append(
            [
                row["decision"],
                fmt_int(row["len"]),
                pct(
                    row["len"]
                    / max(total_migrations, 1)
                ),
            ]
        )

    # ------------------------------------------------------------
    # Top observed migrations
    # ------------------------------------------------------------

    top_rows = []

    top_columns = [
        column
        for column in [
            "repo_name",
            "old_version",
            "new_version",
            duration_column,
            files_column,
            lines_column,
        ]
        if column is not None
        and column in features.columns
    ]

    if duration_column in features.columns:

        top = (
            features
            .sort(
                duration_column,
                descending=True,
                nulls_last=True,
            )
            .head(10)
        )

        for row in top.to_dicts():

            top_rows.append(
                [
                    safe_value(
                        row,
                        "repo_name",
                    ),
                    safe_value(
                        row,
                        "old_version",
                    ),
                    safe_value(
                        row,
                        "new_version",
                    ),
                    fmt_number(
                        safe_value(
                            row,
                            duration_column,
                        )
                    ),
                    fmt_int(
                        safe_value(
                            row,
                            files_column,
                        )
                    ),
                    fmt_int(
                        safe_value(
                            row,
                            lines_column,
                        )
                    ),
                ]
            )

    # ------------------------------------------------------------
    # Risk model evaluation
    # ------------------------------------------------------------

    evaluation_html = """
    <p class="muted">
        Risk-model evaluation is not available yet.
    </p>
    """

    if EVALUATION_PATH.exists():

        evaluation = pl.read_parquet(
            EVALUATION_PATH
        )

        eval_rows = []

        for row in evaluation.to_dicts():

            eval_rows.append(
                [
                    row.get("model"),
                    (
                        f"{row['roc_auc']:.3f}"
                        if row.get("roc_auc") is not None
                        else "N/A"
                    ),
                    (
                        f"{row['pr_auc']:.3f}"
                        if row.get("pr_auc") is not None
                        else "N/A"
                    ),
                    (
                        f"{row['brier_score']:.3f}"
                        if row.get("brier_score") is not None
                        else "N/A"
                    ),
                    (
                        f"{row['f1']:.3f}"
                        if row.get("f1") is not None
                        else "N/A"
                    ),
                ]
            )

        evaluation_html = table(
            [
                "Model",
                "ROC-AUC",
                "PR-AUC",
                "Brier Score",
                "F1",
            ],
            eval_rows,
        )

    # ------------------------------------------------------------
    # Recent migrations
    # ------------------------------------------------------------

    recent_html = ""

    if "migration_timestamp" in features.columns:

        recent = (
            features
            .sort(
                "migration_timestamp",
                descending=True,
                nulls_last=True,
            )
            .head(10)
        )

        recent_rows = []

        for row in recent.to_dicts():

            recent_rows.append(
                [
                    safe_value(
                        row,
                        "migration_timestamp",
                    ),
                    safe_value(
                        row,
                        "repo_name",
                    ),
                    safe_value(
                        row,
                        "old_version",
                    ),
                    safe_value(
                        row,
                        "new_version",
                    ),
                ]
            )

        recent_html = table(
            [
                "Timestamp",
                "Repository",
                "From",
                "To",
            ],
            recent_rows,
        )

    # ------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------

    recommendations = [
        (
            "Use repository history",
            "Review previous migrations before estimating effort, "
            "especially where repeated or large migrations are observed."
        ),
        (
            "Validate compatibility",
            "Prioritize Java compatibility and dependency validation "
            "when version distance is large."
        ),
        (
            "Stage large changes",
            "Use incremental changes and focused regression testing "
            "for migrations with historically high change volume."
        ),
        (
            "Treat risk as evidence",
            "MIGRO provides observed evidence and model estimates, "
            "not guarantees of migration failure or success."
        ),
    ]

    recommendation_html = ""

    for title, description in recommendations:

        recommendation_html += f"""
        <div class="recommendation">
            <div class="recommendation-title">
                {escape(title)}
            </div>
            <div class="recommendation-text">
                {escape(description)}
            </div>
        </div>
        """

    # ------------------------------------------------------------
    # HTML
    # ------------------------------------------------------------

    html = f"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>MIGRO - Migration Intelligence</title>

<style>

:root {{
    --bg: #07090d;
    --panel: #0d1118;
    --panel2: #111722;
    --border: #202938;
    --text: #edf2f7;
    --muted: #8d98a8;
    --accent: #5ee7ff;
    --accent2: #8b7cff;
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background:
        radial-gradient(
            circle at top right,
            rgba(94,231,255,0.08),
            transparent 35%
        ),
        radial-gradient(
            circle at top left,
            rgba(139,124,255,0.08),
            transparent 35%
        ),
        var(--bg);

    color: var(--text);

    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    line-height: 1.5;
}}

.container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 48px 32px 80px;
}}

.header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 30px;
    margin-bottom: 48px;
}}

.brand {{
    font-size: 14px;
    letter-spacing: 0.18em;
    color: var(--accent);
    font-weight: 700;
}}

h1 {{
    font-size: clamp(38px, 5vw, 68px);
    line-height: 0.98;
    margin: 12px 0 18px;
    letter-spacing: -0.045em;
}}

.subtitle {{
    max-width: 700px;
    color: var(--muted);
    font-size: 17px;
}}

.meta {{
    color: var(--muted);
    font-size: 13px;
    text-align: right;
}}

.section {{
    margin-top: 54px;
}}

.section-title {{
    font-size: 23px;
    margin-bottom: 8px;
}}

.section-description {{
    color: var(--muted);
    margin-bottom: 22px;
}}

.kpi-grid {{
    display: grid;
    grid-template-columns:
        repeat(4, minmax(0, 1fr));
    gap: 16px;
}}

.kpi-card {{
    background:
        linear-gradient(
            145deg,
            var(--panel2),
            var(--panel)
        );

    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
}}

.kpi-title {{
    color: var(--muted);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.kpi-value {{
    font-size: 34px;
    font-weight: 700;
    margin-top: 8px;
}}

.kpi-subtitle {{
    color: var(--muted);
    font-size: 12px;
    margin-top: 5px;
}}

.grid-2 {{
    display: grid;
    grid-template-columns:
        repeat(2, minmax(0, 1fr));
    gap: 20px;
}}

.panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
}}

.panel h3 {{
    margin-top: 0;
    margin-bottom: 18px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

th {{
    color: var(--muted);
    font-weight: 500;
    text-align: left;
    border-bottom: 1px solid var(--border);
    padding: 11px 10px;
}}

td {{
    border-bottom: 1px solid #18202c;
    padding: 11px 10px;
}}

tr:last-child td {{
    border-bottom: none;
}}

.recommendations {{
    display: grid;
    grid-template-columns:
        repeat(2, minmax(0, 1fr));
    gap: 16px;
}}

.recommendation {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px;
}}

.recommendation-title {{
    font-weight: 700;
    margin-bottom: 8px;
}}

.recommendation-text {{
    color: var(--muted);
    font-size: 14px;
}}

.architecture {{
    font-family:
        "SFMono-Regular",
        Consolas,
        monospace;

    background: #05070a;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    overflow-x: auto;
    color: #b7c2d1;
    font-size: 13px;
    line-height: 1.8;
}}

.note {{
    color: var(--muted);
    font-size: 13px;
}}

.muted {{
    color: var(--muted);
}}

.footer {{
    margin-top: 70px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 12px;
}}

@media (max-width: 900px) {{

    .kpi-grid {{
        grid-template-columns:
            repeat(2, minmax(0, 1fr));
    }}

    .grid-2,
    .recommendations {{
        grid-template-columns: 1fr;
    }}

    .header {{
        display: block;
    }}

    .meta {{
        text-align: left;
        margin-top: 20px;
    }}
}}

@media (max-width: 600px) {{

    .container {{
        padding: 28px 18px 60px;
    }}

    .kpi-grid {{
        grid-template-columns: 1fr;
    }}
}}

</style>

</head>

<body>

<div class="container">

<header class="header">

<div>

<div class="brand">
MIGRO / SOFTWARE MIGRATION INTELLIGENCE
</div>

<h1>
Migration intelligence<br>
from historical evidence.
</h1>

<div class="subtitle">
MIGRO analyzes open-source Java migration activity to
understand migration effort, change structure, historical
repository behavior, anomalies, and evidence-informed
engineering preparation.
</div>

</div>

<div class="meta">
CoUpJava-Coarse dataset<br>
{total_migrations:,} observed migrations<br>
Generated from local MIGRO analysis artifacts
</div>

</header>


<section class="section">

<div class="section-title">
Executive Snapshot
</div>

<div class="section-description">
A high-level view of the observed migration landscape.
</div>

<div class="kpi-grid">

{card(
    "Observed migrations",
    fmt_int(total_migrations),
    "Migration events analyzed"
)}

{card(
    "Median duration",
    f"{fmt_number(duration_median)} days",
    "Typical observed migration interval"
)}

{card(
    "P75 duration",
    f"{fmt_number(duration_p75)} days",
    "Upper-quartile migration interval"
)}

{card(
    "High-duration share",
    pct(high_duration_rate),
    "Migrations at or above P75"
)}

</div>

</section>


<section class="section">

<div class="kpi-grid">

{card(
    "Median files changed",
    fmt_int(files_median),
    "Observed migration change volume"
)}

{card(
    "Median changed lines",
    fmt_int(lines_median),
    "Observed line-change volume"
)}

{card(
    "Evidence records",
    fmt_int(evidence.height),
    "Migration-context assessments"
)}

{card(
    "Decision records",
    fmt_int(decisions.height),
    "Engineering decision assessments"
)}

</div>

</section>


<section class="section">

<div class="section-title">
Evidence Landscape
</div>

<div class="section-description">
How much observable migration-history evidence exists for
the analyzed repository events.
</div>

<div class="grid-2">

<div class="panel">

<h3>Evidence levels</h3>

{table(
    [
        "Evidence level",
        "Migrations",
        "Share",
    ],
    evidence_rows,
)}

</div>


<div class="panel">

<h3>Decision distribution</h3>

{table(
    [
        "Decision",
        "Migrations",
        "Share",
    ],
    decision_rows,
)}

</div>

</div>

</section>


<section class="section">

<div class="section-title">
Highest Observed Migration Effort
</div>

<div class="section-description">
Retrospective view of migrations with the longest observed
migration intervals. These are observations, not predictions.
</div>

<div class="panel">

{table(
    [
        "Repository",
        "From",
        "To",
        "Duration (days)",
        "Files",
        "Changed lines",
    ],
    top_rows,
)}

</div>

</section>


<section class="section">

<div class="section-title">
Risk Model Evaluation
</div>

<div class="section-description">
Comparison of simple baselines with MIGRO's historical-context
model. Scores are evaluation metrics, not guarantees of future
migration outcomes.
</div>

<div class="panel">

{evaluation_html}

</div>

</section>


<section class="section">

<div class="section-title">
Recent Migration Activity
</div>

<div class="section-description">
Latest migration events represented in the dataset.
</div>

<div class="panel">

{recent_html}

</div>

</section>


<section class="section">

<div class="section-title">
Engineering Decision Guidance
</div>

<div class="section-description">
Evidence-informed preparation actions generated by MIGRO.
</div>

<div class="recommendations">

{recommendation_html}

</div>

</section>


<section class="section">

<div class="section-title">
MIGRO Architecture
</div>

<div class="section-description">
The analytical path from raw migration history to engineering
decision support.
</div>

<div class="architecture">

Raw migration history
        |
        v
Migration profiling
        |
        +---- Change size
        +---- Dependency signals
        +---- Change coupling
        +---- Temporal behavior
        +---- Version benchmarks
        +---- Anomalies
        |
        v
Historical repository baselines
        |
        v
Leakage-controlled risk model
        |
        v
Risk evidence
        |
        v
Decision engine
        |
        v
Engineering preparation guidance

</div>

</section>


<section class="section">

<div class="section-title">
Methodology & Limitations
</div>

<div class="panel">

<p class="note">
<strong>Dataset:</strong>
MIGRO currently analyzes the CoUpJava-Coarse open-source
migration dataset.
</p>

<p class="note">
<strong>Risk model:</strong>
The predictive layer uses historical repository characteristics
and Java version distance. The evaluation is chronological rather
than random, reducing the risk of using later migrations to
evaluate earlier ones.
</p>

<p class="note">
<strong>Interpretation:</strong>
Observed associations and model coefficients do not establish
causation. Evidence levels are descriptive indicators rather than
probabilities of migration failure.
</p>

<p class="note">
<strong>Migration context:</strong>
Some analytical layers use information observed in the migration
patch itself. Those features are intentionally separated from the
pre-migration predictive feature set to reduce leakage.
</p>

<p class="note">
<strong>Scope:</strong>
Results describe the observed dataset and should not automatically
be generalized to every Java codebase or migration environment.
</p>

</div>

</section>


<footer class="footer">

MIGRO — Open-source software migration intelligence.
Built as a data-to-decision engineering system.

</footer>

</div>

</body>

</html>
"""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        html,
        encoding="utf-8",
    )

    print(
        f"\nDashboard generated:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        f"\nSize: "
        f"{OUTPUT_PATH.stat().st_size:,} bytes"
    )

    print("\n" + "=" * 70)
    print("MIGRO dashboard generation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
