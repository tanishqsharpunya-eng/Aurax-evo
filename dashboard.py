"""
AURAX Evo — Evolution Dashboard (dashboard.py)
================================================
Real-time Streamlit dashboard for monitoring the self-improvement loop.

Run with:
    streamlit run dashboard.py -- --config config_evo.yaml

Features:
  - Live generation history table
  - Score progression chart
  - Per-benchmark radar chart
  - Curriculum weight evolution
  - Safety violation alerts
  - Model promotion timeline
"""

import json
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AURAX Evo Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark industrial theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Space+Grotesk:wght@300;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Dark background */
.stApp { background: #0a0c10; }
.stSidebar { background: #0e1117 !important; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #141820, #1c2230);
    border: 1px solid #2a3347;
    border-radius: 10px;
    padding: 18px 24px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a7a9a;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #e0f0ff;
    font-family: 'JetBrains Mono', monospace;
}
.metric-delta-pos { color: #3de88e; font-size: 14px; }
.metric-delta-neg { color: #e85b3d; font-size: 14px; }

/* Status badges */
.badge-pass { background: #0d2d1a; color: #3de88e; border: 1px solid #1e6640; 
              padding: 2px 10px; border-radius: 20px; font-size: 12px; }
.badge-fail { background: #2d0d0d; color: #e85b3d; border: 1px solid #661e1e; 
              padding: 2px 10px; border-radius: 20px; font-size: 12px; }
.badge-warn { background: #2d220d; color: #e8b83d; border: 1px solid #664e1e; 
              padding: 2px 10px; border-radius: 20px; font-size: 12px; }

/* Section headers */
h1 { color: #c0d8f0 !important; letter-spacing: -1px; }
h2 { color: #90b8d8 !important; font-size: 16px !important; letter-spacing: 1px; }
h3 { color: #6090b0 !important; font-size: 13px !important; }

/* Table */
.stDataFrame { border: 1px solid #2a3347 !important; border-radius: 8px; }

/* Alert */
.alert-box {
    background: #2d0d0d;
    border-left: 4px solid #e85b3d;
    padding: 12px 18px;
    border-radius: 6px;
    margin: 12px 0;
    color: #f0a0a0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=5)   # refresh every 5 seconds
def load_history(history_path: str) -> list[dict]:
    p = Path(history_path)
    if not p.exists():
        return []
    records = []
    for line in p.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


@st.cache_data(ttl=5)
def load_curriculum_state(state_path: str) -> dict:
    p = Path(state_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    import yaml
    try:
        return yaml.safe_load(p.read_text())
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ AURAX Evo")
    st.markdown("---")

    config_path = st.text_input("Config file", value="config_evo.yaml")
    config = load_config(config_path)

    history_path = config.get("evolution", {}).get("history_path", "evo_history.jsonl")
    curriculum_path = config.get("evolution", {}).get("curriculum_state", "curriculum_state.json")

    st.markdown(f"**History:** `{history_path}`")
    st.markdown(f"**Curriculum:** `{curriculum_path}`")
    st.markdown("---")

    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
    if auto_refresh:
        refresh_rate = st.slider("Refresh interval (s)", 2, 30, 5)

    st.markdown("---")
    st.markdown("**Loop config:**")
    gen_cfg = config.get("generation", {})
    st.markdown(f"- Max generations: `{gen_cfg.get('max_generations', 50)}`")
    st.markdown(f"- Samples/gen: `{gen_cfg.get('samples_per_generation', 5000)}`")
    st.markdown(f"- Threshold: `{gen_cfg.get('improvement_threshold', 0.05):.0%}`")
    st.markdown(f"- Approval every: `{gen_cfg.get('human_approval_every', 5)}`")

    st.markdown("---")
    if st.button("🔄 Force Refresh"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown("# 🧬 AURAX Evo — Self-Improvement Monitor")

records = load_history(history_path)
curriculum_state = load_curriculum_state(curriculum_path)

if not records:
    st.info("No evolution history yet. Start the loop with `python evo_loop.py`")
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------------------
latest = records[-1]
promoted_records = [r for r in records if r.get("promoted", False)]
safety_failures = [r for r in records if not r.get("safety_passed", True)]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Generations Run</div>
      <div class="metric-value">{len(records)}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    best_score = max((r.get("candidate_score", 0) for r in promoted_records), default=0)
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Best Score</div>
      <div class="metric-value">{best_score:.4f}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    promo_rate = len(promoted_records) / len(records) * 100 if records else 0
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Promotion Rate</div>
      <div class="metric-value">{promo_rate:.0f}%</div>
    </div>""", unsafe_allow_html=True)

with col4:
    safety_label = f'<span class="badge-fail">⚠ {len(safety_failures)} fail</span>' if safety_failures \
                   else '<span class="badge-pass">✓ All pass</span>'
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Safety Status</div>
      <div class="metric-value" style="font-size:18px;padding-top:8px;">{safety_label}</div>
    </div>""", unsafe_allow_html=True)

with col5:
    current_model = latest.get("current_model", "—")
    short_model = Path(current_model).name if current_model != "—" else "—"
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Current Model</div>
      <div class="metric-value" style="font-size:14px;padding-top:8px;">{short_model}</div>
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Safety alerts
# ---------------------------------------------------------------------------
if safety_failures:
    st.markdown(f"""
    <div class="alert-box">
      ⚠️ <strong>{len(safety_failures)} safety violation(s)</strong> detected in generations:
      {', '.join(str(r['generation']) for r in safety_failures)}
    </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Score progression chart
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📈 Score Progression")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    gens = [r["generation"] for r in records]
    current_scores = [r.get("current_score", 0) for r in records]
    candidate_scores = [r.get("candidate_score", 0) for r in records]
    promotions = [r["generation"] for r in records if r.get("promoted")]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gens, y=current_scores,
        name="Current Model",
        line=dict(color="#4a90d9", width=2),
        mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=gens, y=candidate_scores,
        name="Candidate Model",
        line=dict(color="#3de88e", width=2, dash="dot"),
        mode="lines+markers",
    ))

    for g in promotions:
        fig.add_vline(
            x=g, line_color="#3de88e", line_dash="dash", opacity=0.4,
            annotation_text="↑", annotation_position="top",
        )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        font=dict(color="#8090a8", family="JetBrains Mono"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8090a8")),
        xaxis=dict(title="Generation", gridcolor="#1a2234"),
        yaxis=dict(title="Overall Score", gridcolor="#1a2234"),
        height=350,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

except ImportError:
    # Fallback to matplotlib
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0a0c10")
    ax.set_facecolor("#0e1117")

    gens = [r["generation"] for r in records]
    ax.plot(gens, [r.get("current_score", 0) for r in records],
            label="Current", color="#4a90d9", lw=2)
    ax.plot(gens, [r.get("candidate_score", 0) for r in records],
            label="Candidate", color="#3de88e", lw=2, ls="--")

    for g in [r["generation"] for r in records if r.get("promoted")]:
        ax.axvline(g, color="#3de88e", alpha=0.4, ls=":")

    ax.set_xlabel("Generation", color="#8090a8")
    ax.set_ylabel("Score", color="#8090a8")
    ax.tick_params(colors="#8090a8")
    ax.legend(facecolor="#0e1117", labelcolor="#8090a8")
    ax.spines[:].set_color("#2a3347")
    plt.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Curriculum weights chart
# ---------------------------------------------------------------------------
st.markdown("---")
col_cur, col_bm = st.columns(2)

with col_cur:
    st.markdown("## 🎓 Curriculum Weights")
    if curriculum_state and curriculum_state.get("history"):
        hist = curriculum_state["history"]
        c_gens = [h["generation"] for h in hist]
        easy_w   = [h["proportions"].get("easy", 0) for h in hist]
        medium_w = [h["proportions"].get("medium", 0) for h in hist]
        hard_w   = [h["proportions"].get("hard", 0) for h in hist]

        try:
            import plotly.graph_objects as go
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=c_gens, y=easy_w,   name="Easy",   stackgroup="one",
                                      fillcolor="rgba(74,144,217,0.5)", line_color="#4a90d9"))
            fig2.add_trace(go.Scatter(x=c_gens, y=medium_w, name="Medium", stackgroup="one",
                                      fillcolor="rgba(230,180,60,0.5)", line_color="#e6b43c"))
            fig2.add_trace(go.Scatter(x=c_gens, y=hard_w,   name="Hard",   stackgroup="one",
                                      fillcolor="rgba(232,91,61,0.5)", line_color="#e85b3d"))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)",
                font=dict(color="#8090a8"), height=280,
                margin=dict(t=10, b=30),
                xaxis=dict(gridcolor="#1a2234"),
                yaxis=dict(gridcolor="#1a2234", range=[0, 1]),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            st.json({
                "easy":   easy_w[-1] if easy_w else "—",
                "medium": medium_w[-1] if medium_w else "—",
                "hard":   hard_w[-1] if hard_w else "—",
            })

        # Tier stats
        tier_stats = curriculum_state.get("tier_stats", {})
        for tier, stats in tier_stats.items():
            att = stats.get("attempts", 0)
            suc = stats.get("successes", 0)
            acc = suc / att * 100 if att > 0 else 0
            st.markdown(f"**{tier.capitalize()}**: {suc}/{att} ({acc:.1f}%)")
    else:
        st.info("No curriculum history yet.")


with col_bm:
    st.markdown("## 🏆 Latest Benchmark Scores")
    bm_details = latest.get("benchmark_details", {})
    if bm_details:
        try:
            import plotly.graph_objects as go
            categories = list(bm_details.keys())
            current_vals  = [bm_details[k].get("current", 0) for k in categories]
            candidate_vals = [bm_details[k].get("candidate", 0) for k in categories]

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(name="Current",   x=categories, y=current_vals,
                                  marker_color="#4a90d9"))
            fig3.add_trace(go.Bar(name="Candidate", x=categories, y=candidate_vals,
                                  marker_color="#3de88e"))
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)",
                font=dict(color="#8090a8"), height=280,
                margin=dict(t=10, b=30),
                xaxis=dict(gridcolor="#1a2234"),
                yaxis=dict(gridcolor="#1a2234", range=[0, 1]),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                barmode="group",
            )
            st.plotly_chart(fig3, use_container_width=True)
        except ImportError:
            for k, v in bm_details.items():
                st.markdown(f"**{k}**: current={v.get('current', 0):.3f} | candidate={v.get('candidate', 0):.3f}")
    else:
        st.info("No benchmark details in latest generation.")


# ---------------------------------------------------------------------------
# History table
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 📋 Generation History")

try:
    import pandas as pd
    df_data = []
    for r in records:
        df_data.append({
            "Gen": r.get("generation", "?"),
            "Current Score": f"{r.get('current_score', 0):.4f}",
            "Candidate Score": f"{r.get('candidate_score', 0):.4f}",
            "Δ%": f"{r.get('delta_pct', 0):+.2f}%",
            "Promoted": "✓" if r.get("promoted") else "✗",
            "Safety": "✓" if r.get("safety_passed") else "⚠️",
            "Samples": r.get("num_samples_generated", 0),
            "Train (s)": f"{r.get('train_duration_s', 0):.0f}",
            "Reason": r.get("reason", ""),
        })
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
except ImportError:
    for r in records[-10:]:
        st.markdown(
            f"Gen {r.get('generation','?')} | "
            f"Score: {r.get('candidate_score', 0):.4f} | "
            f"{'✓ PROMOTED' if r.get('promoted') else '✗'} | "
            f"{r.get('reason', '')}"
        )


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
