"""
APE-BAIT Interactive — Streamlit Frontend

Run with:
    streamlit run src/frontend/app.py
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

API_BASE = os.getenv("APE_BAIT_API_URL", "http://localhost:8000/api/v1")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="APE-BAIT Interactive",
    page_icon="[APE-BAIT]",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #000;
    color: #e2f1e6;
  }

  .stApp { background: #000000; }

  h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; }
  h1 { color: #00ff66; letter-spacing: -0.5px; }
  h2 { color: #e2f1e6; }
  h3 { color: #00ff66; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px; }

  .brand-header {
    border-bottom: 1px solid #1f2922;
    padding-bottom: 14px;
    margin-bottom: 24px;
  }
  .brand-name {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 26px;
    color: #00ff66;
    letter-spacing: -0.5px;
  }
  .brand-sub {
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    color: #4a5c51;
    margin-top: 4px;
  }

  .metric-card {
    background: #0a0a0a;
    border: 1px solid #1f2922;
    border-radius: 6px;
    padding: 16px;
    text-align: center;
  }
  .metric-card .mval { font-family: 'Fira Code'; font-size: 28px; font-weight: 600; color: #00ff66; }
  .metric-card .mlabel { font-size: 11px; color: #4a5c51; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.8px; }
  .metric-card .mstatus { font-family: 'Fira Code'; font-size: 10px; margin-top: 6px; }
  .mstatus.pass { color: #00ff66; } .mstatus.fail { color: #ffaa00; }

  .tool-row {
    display: flex; align-items: center; justify-content: space-between;
    background: #0a0a0a; border: 1px solid #1f2922; border-radius: 4px;
    padding: 9px 14px; margin-bottom: 6px; font-family: 'Fira Code', monospace; font-size: 12px;
  }
  .evade-bar-wrap { display: flex; align-items: center; gap: 10px; }
  .evade-bar { height: 5px; background: #1f2922; border-radius: 3px; flex:1; }
  .evade-fill { height: 100%; border-radius: 3px; }

  .log-box {
    background: #050505; border: 1px solid #1f2922; border-radius: 4px;
    padding: 12px; font-family: 'Fira Code', monospace; font-size: 11.5px;
    color: #4a5c51; max-height: 200px; overflow-y: auto;
  }

  /* Streamlit component overrides */
  .stSlider > div > div > div { background: #1f2922 !important; }
  .stSelectbox > div > div { background: #0a0a0a !important; border-color: #1f2922 !important; }
  .stButton > button {
    background: #00ff66 !important; color: #000 !important; border: none !important;
    font-weight: 700 !important; font-family: 'Inter', sans-serif !important;
    border-radius: 4px !important; padding: 8px 24px !important;
  }
  .stButton > button:hover { background: #39ff14 !important; }
  div[data-testid="stSidebar"] { background: #0a0a0a !important; border-right: 1px solid #1f2922; }
  .stProgress > div > div > div { background: #00ff66 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str) -> dict | list | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, **kwargs) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=10, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def make_evasion_chart(tool_results: list[dict]) -> go.Figure:
    df = pd.DataFrame(tool_results)
    df["pct"] = (df["evasion_rate"] * 100).round(1)
    df = df.sort_values("pct", ascending=True)
    colors = ["#00ff66" if r >= 75 else "#ffaa00" if r >= 50 else "#ff4444" for r in df["pct"]]
    fig = go.Figure(go.Bar(
        x=df["pct"], y=df["tool"],
        orientation="h",
        marker_color=colors,
        text=[f"{v}%" for v in df["pct"]],
        textposition="inside",
        insidetextanchor="start",
    ))
    fig.update_layout(
        paper_bgcolor="#000", plot_bgcolor="#000",
        font=dict(family="Fira Code", color="#89a391", size=11),
        xaxis=dict(range=[0, 100], gridcolor="#1f2922", title="Evasion Rate (%)"),
        yaxis=dict(gridcolor="#1f2922"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(280, len(df) * 30),
    )
    return fig


def make_latency_histogram(tool_results: list[dict]) -> go.Figure:
    times = [r["execution_time_s"] * 1000 for r in tool_results]
    fig = go.Figure(go.Bar(
        x=[r["tool"] for r in tool_results],
        y=times,
        marker_color="#00ff66",
    ))
    fig.add_hline(y=2000, line_dash="dash", line_color="#ffaa00", annotation_text="2s budget")
    fig.update_layout(
        paper_bgcolor="#000", plot_bgcolor="#000",
        font=dict(family="Fira Code", color="#89a391", size=11),
        xaxis=dict(gridcolor="#1f2922"),
        yaxis=dict(gridcolor="#1f2922", title="Latency (ms)"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=250,
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-name">APE-BAIT</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">adversarial perturbation engine</div>', unsafe_allow_html=True)
    st.markdown("---")

    nav = st.radio(
        "Navigate",
        ["New Job", "Job History", "API Docs"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # API health
    health = api_get("/health")
    if health and health.get("status") == "ok":
        st.success("API connected", icon=None)
    else:
        st.warning("API offline — start with `uvicorn src.api.main:app`")

    st.markdown("""
    <div style="font-family:Fira Code; font-size:10px; color:#4a5c51; margin-top:16px;">
    v1.0.0 &middot; python / fastapi / celery<br>
    pytorch &middot; scapy &middot; art
    </div>
    """, unsafe_allow_html=True)


# ── View: New Job ─────────────────────────────────────────────────────────────
if nav == "New Job":
    st.markdown('<h1>New Perturbation Job</h1>', unsafe_allow_html=True)
    st.markdown(
        "Upload PCAP/CSV traffic or use the built-in sample, tune attack parameters, "
        "select target security tools, and run the adversarial engine.",
        unsafe_allow_html=False,
    )

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown("### Traffic Source")
        source = st.selectbox("Source", ["Upload PCAP/CSV", "Use sample traffic", "Live capture"], label_visibility="collapsed")

        uploaded_file = None
        live_iface = None
        capture_dur = 30

        if source == "Upload PCAP/CSV":
            uploaded_file = st.file_uploader("Upload PCAP or CSV", type=["pcap", "pcapng", "csv"])
        elif source == "Live capture":
            live_iface = st.text_input("Network interface", value="eth0")
            capture_dur = st.slider("Capture duration (s)", 5, 120, 30)
        else:
            st.info("Will use `data/sample_pcaps/sample_traffic.pcap`")

        st.markdown("### Attack Method")
        method = st.selectbox("Method", ["PGD (recommended)", "FGSM (fast)", "Ensemble"])
        method_id = {"PGD (recommended)": "pgd", "FGSM (fast)": "fgsm", "Ensemble": "ensemble"}[method]

        st.markdown("### Parameters")
        epsilon = st.slider("Epsilon (perturbation budget)", 0.001, 0.1, 0.03, 0.001,
                            help="L-inf norm bound. Stay below 0.04 for human-invisible perturbations.")
        alpha = st.slider("Alpha (step size, PGD only)", 0.001, 0.05, 0.005, 0.001)
        iterations = st.slider("Iterations (PGD only)", 1, 100, 10)

        st.caption(f"MSE budget: {'within 0.04 threshold' if epsilon <= 0.04 else 'WARNING: may exceed 0.04 MSE'}")

    with col_right:
        st.markdown("### Target Security Tools")

        # Fetch tool catalog
        tool_catalog = api_get("/results/tools") or []
        tool_options = {t["name"]: t["id"] for t in tool_catalog} if tool_catalog else {
            "Snort": "snort", "Suricata": "suricata", "YARA": "yara",
            "MalConv": "malconv", "CNN IDS": "cnn_ids", "Random Forest IDS": "random_forest",
            "LSTM IDS": "lstm_ids", "Isolation Forest": "isolation_forest",
            "Autoencoder": "autoencoder", "EMBER LightGBM": "ember_gbm",
        }

        select_all = st.checkbox("Select all tools", value=True)
        if select_all:
            selected_names = list(tool_options.keys())
        else:
            selected_names = st.multiselect("Tools", list(tool_options.keys()), default=list(tool_options.keys())[:5])

        selected_ids = [tool_options[n] for n in selected_names]

        st.markdown(f"""
        <div class="log-box">
        Selected: {len(selected_ids)} tools<br>
        Method: {method_id.upper()} &nbsp;|&nbsp; epsilon={epsilon} &nbsp;|&nbsp;
        alpha={alpha if method_id != 'fgsm' else 'N/A'} &nbsp;|&nbsp;
        iters={iterations if method_id != 'fgsm' else 1}<br>
        Evasion expected: ~{'80-92%' if method_id in ('pgd','ensemble') else '70-82%'}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    run_col, _ = st.columns([0.2, 0.8])
    with run_col:
        run_btn = st.button("Run Attack", use_container_width=True)

    # ── Job submission ────────────────────────────────────────────────────────
    if run_btn:
        if source == "Upload PCAP/CSV" and not uploaded_file:
            st.error("Please upload a PCAP or CSV file.")
        else:
            with st.spinner("Submitting job..."):
                form_data = {
                    "method": method_id,
                    "epsilon": str(epsilon),
                    "alpha": str(alpha),
                    "iterations": str(iterations),
                    "target_tools": ",".join(selected_ids),
                }

                if source == "Upload PCAP/CSV":
                    form_data["traffic_source"] = "upload"
                    files = {"pcap_file": (uploaded_file.name, uploaded_file.read(), "application/octet-stream")}
                    result = api_post("/jobs/", data=form_data, files=files)
                elif source == "Use sample traffic":
                    form_data["traffic_source"] = "sample"
                    result = api_post("/jobs/", data=form_data)
                else:
                    form_data["traffic_source"] = "live"
                    form_data["live_interface"] = live_iface
                    form_data["capture_duration_s"] = str(capture_dur)
                    result = api_post("/jobs/", data=form_data)

            if result:
                st.session_state["active_job_id"] = result["job_id"]
                st.success(f"Job submitted: `{result['job_id']}`")

    # ── Live progress ─────────────────────────────────────────────────────────
    if "active_job_id" in st.session_state:
        job_id = st.session_state["active_job_id"]
        st.markdown("---")
        st.markdown(f"### Results — Job `{job_id[:8]}...`")

        job = api_get(f"/jobs/{job_id}")
        if not job:
            st.error("Could not fetch job status.")
        else:
            status = job["status"]
            progress = job.get("progress", 0)
            msg = job.get("progress_message", "")

            if status in ("pending", "running"):
                st.progress(progress / 100, text=f"{msg} ({progress}%)")
                time.sleep(1)
                st.rerun()

            elif status == "success":
                summary = job.get("summary", {})
                tool_results = job.get("tool_results", [])

                # KPI row
                k1, k2, k3, k4 = st.columns(4)
                evasion_pct = f"{summary.get('aggregate_evasion_rate', 0) * 100:.1f}%"
                mse = summary.get("distortion", {}).get("mean_mse", 0)
                latency = summary.get("latency", {}).get("p95_ms", 0)
                evaded = f"{summary.get('tools_evaded', 0)}/{summary.get('tools_tested', 0)}"

                for col, val, label, sub, ok in [
                    (k1, evasion_pct, "Evasion Rate", ">80% target", float(evasion_pct.strip("%")) >= 80),
                    (k2, f"{mse:.4f}", "MSE Distortion", "<0.04 threshold", mse < 0.04),
                    (k3, f"{latency:.0f}ms", "P95 Latency", "<2000ms budget", latency < 2000),
                    (k4, evaded, "Tools Evaded", "of tools tested", True),
                ]:
                    with col:
                        status_class = "pass" if ok else "fail"
                        st.markdown(f"""
                        <div class="metric-card">
                          <div class="mval">{val}</div>
                          <div class="mlabel">{label}</div>
                          <div class="mstatus {status_class}">{sub}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("#### Evasion rate by tool")
                if tool_results:
                    st.plotly_chart(make_evasion_chart(tool_results), use_container_width=True)
                    st.plotly_chart(make_latency_histogram(tool_results), use_container_width=True)

                    # Download buttons
                    dl_col1, dl_col2 = st.columns(2)
                    with dl_col1:
                        pcap_bytes = requests.get(f"{API_BASE}/jobs/{job_id}/pcap").content
                        st.download_button("Download Perturbed PCAP", pcap_bytes,
                                           file_name=f"ape-bait-{job_id[:8]}.pcap",
                                           mime="application/octet-stream")
                    with dl_col2:
                        report_bytes = requests.get(f"{API_BASE}/jobs/{job_id}/report").content
                        st.download_button("Download JSON Report", report_bytes,
                                           file_name=f"ape-bait-report-{job_id[:8]}.json",
                                           mime="application/json")

            elif status == "failed":
                st.error(f"Job failed: {job.get('error', 'Unknown error')}")


# ── View: Job History ─────────────────────────────────────────────────────────
elif nav == "Job History":
    st.markdown('<h1>Job History</h1>', unsafe_allow_html=True)

    jobs_data = api_get("/jobs/")
    if not jobs_data:
        st.info("No jobs yet. Run a perturbation job first.")
    else:
        jobs = jobs_data.get("jobs", [])
        if not jobs:
            st.info("No jobs yet.")
        else:
            df = pd.DataFrame([{
                "Job ID": j["job_id"][:8] + "...",
                "Status": j["status"],
                "Method": j["attack_params"]["method"].upper(),
                "Epsilon": j["attack_params"]["epsilon"],
                "Evasion": f"{j['summary']['aggregate_evasion_rate'] * 100:.1f}%" if j.get("summary") else "-",
                "MSE": f"{j['summary']['distortion']['mean_mse']:.4f}" if j.get("summary") else "-",
                "Created": j["created_at"][:19],
            } for j in jobs])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Click to load a job
            selected_id = st.text_input("Paste full Job ID to inspect")
            if selected_id:
                st.session_state["active_job_id"] = selected_id
                st.rerun()


# ── View: API Docs ────────────────────────────────────────────────────────────
elif nav == "API Docs":
    st.markdown('<h1>REST API</h1>', unsafe_allow_html=True)
    st.markdown("""
The APE-BAIT FastAPI backend exposes a full REST API for automation and CI/CD integration.

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/jobs/` | POST | Create a new perturbation job (upload PCAP, sample, or live) |
| `/api/v1/jobs/` | GET | List all jobs (paginated) |
| `/api/v1/jobs/{job_id}` | GET | Get job status and results |
| `/api/v1/jobs/{job_id}` | DELETE | Cancel a job |
| `/api/v1/jobs/{job_id}/pcap` | GET | Download perturbed PCAP |
| `/api/v1/jobs/{job_id}/report` | GET | Download JSON validation report |
| `/api/v1/results/tools` | GET | List all supported target security tools |
| `/api/v1/health` | GET | Health check |

Interactive Swagger docs: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)

```bash
# Example: submit a job using curl
curl -X POST http://localhost:8000/api/v1/jobs/ \\
  -F "traffic_source=sample" \\
  -F "method=pgd" \\
  -F "epsilon=0.03" \\
  -F "iterations=10" \\
  -F "target_tools=all"
```
""")
