#!/usr/bin/env python
"""
APE-BAIT Report Generator
Generates an HTML report from a JSON validation results file.

Usage:
    python scripts/generate_report.py --input results/validation_report_*.json
    python scripts/generate_report.py --input report.json --output report.html
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path
from jinja2 import Template


REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>APE-BAIT Validation Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap');
  :root {
    --bg: #050810; --card: #0d1527; --cyan: #00f5ff; --green: #39ff14;
    --orange: #ff6b35; --red: #ff3464; --violet: #bf5fff; --text: #e8f4fd; --muted: #8ba3c4;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text);
         padding: 40px; max-width: 1200px; margin: 0 auto; }
  h1 { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, var(--cyan), var(--violet));
       -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 40px; }
  .meta { font-family: 'JetBrains Mono'; font-size: 0.75rem; color: var(--muted); }
  .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 32px; }
  .kpi { background: var(--card); border-radius: 12px; padding: 20px; border: 1px solid rgba(0,245,255,0.08); }
  .kpi-val { font-family: 'JetBrains Mono'; font-size: 1.8rem; font-weight: 700; }
  .kpi-lbl { color: var(--muted); font-size: 0.75rem; margin-top: 4px; }
  .section { margin-bottom: 32px; }
  .section-title { font-size: 1rem; font-weight: 700; color: var(--cyan); border-bottom: 1px solid rgba(0,245,255,0.1);
                   padding-bottom: 8px; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 12px; overflow: hidden; }
  th { padding: 10px 16px; text-align: left; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px;
       color: var(--muted); background: rgba(0,0,0,0.3); }
  td { padding: 10px 16px; border-bottom: 1px solid rgba(255,255,255,0.03); font-size: 0.82rem; }
  tr:last-child td { border-bottom: none; }
  .bar-wrap { display: flex; align-items: center; gap: 8px; }
  .bar { flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; }
  .bar-fill { height: 100%; border-radius: 3px; }
  .pct { font-family: 'JetBrains Mono'; font-size: 0.78rem; font-weight: 600; min-width: 42px; text-align: right; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.65rem; font-family: 'JetBrains Mono'; }
  .badge-green { background: rgba(57,255,20,0.1); color: var(--green); border: 1px solid rgba(57,255,20,0.2); }
  .badge-orange { background: rgba(255,107,53,0.1); color: var(--orange); border: 1px solid rgba(255,107,53,0.2); }
  .badge-red { background: rgba(255,52,100,0.1); color: var(--red); border: 1px solid rgba(255,52,100,0.2); }
  footer { margin-top: 48px; text-align: center; color: var(--muted); font-size: 0.75rem; font-family: 'JetBrains Mono'; }
</style>
</head>
<body>

<h1>🦍⚡ APE-BAIT Validation Report</h1>
<p class="subtitle">Adversarial Perturbation Engine — Security Tool Evasion Results</p>
<p class="meta">
  Generated: {{ timestamp }} &nbsp;|&nbsp;
  Method: {{ metadata.method | upper }} &nbsp;|&nbsp;
  ε = {{ metadata.epsilon }} &nbsp;|&nbsp;
  PCAP: {{ metadata.input_pcap }}
</p>

<!-- KPIs -->
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-val" style="color:var(--cyan)">{{ "%.1f" | format(summary.aggregate_evasion_rate * 100) }}%</div>
    <div class="kpi-lbl">Aggregate Evasion Rate</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:{{ 'var(--green)' if summary.mse_below_threshold else 'var(--red)' }}">
      {{ "%.4f" | format(summary.avg_mse) }}
    </div>
    <div class="kpi-lbl">Avg MSE Distortion</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:var(--violet)">{{ "%.0f" | format(summary.avg_latency_ms) }}ms</div>
    <div class="kpi-lbl">Avg Injection Latency</div>
  </div>
  <div class="kpi">
    <div class="kpi-val" style="color:var(--orange)">{{ summary.tools_evaded }}/{{ summary.tools_tested }}</div>
    <div class="kpi-lbl">Tools Evaded (≥50%)</div>
  </div>
</div>

<!-- Tool Results Table -->
<div class="section">
  <div class="section-title">Security Tool Evasion Results</div>
  <table>
    <thead>
      <tr>
        <th>Tool</th>
        <th>Original Detections</th>
        <th>Perturbed Detections</th>
        <th>Evasion Rate</th>
        <th>Status</th>
        <th>Time (s)</th>
      </tr>
    </thead>
    <tbody>
      {% for tool_name, result in evasion_by_tool.items() %}
      {% set rate = result.evasion_rate %}
      {% set color = '#39ff14' if rate >= 0.85 else ('#00f5ff' if rate >= 0.75 else '#ff6b35') %}
      {% set badge_class = 'badge-green' if rate >= 0.75 else 'badge-orange' %}
      {% set status = '✅ Evaded' if rate >= 0.5 else '⚠️ Partial' %}
      <tr>
        <td><strong>{{ tool_name }}</strong>
          {% if result.error %}<span class="badge badge-red">ERROR</span>{% endif %}
        </td>
        <td style="font-family:JetBrains Mono">{{ result.original_detections }}</td>
        <td style="font-family:JetBrains Mono">{{ result.perturbed_detections }}</td>
        <td>
          <div class="bar-wrap">
            <div class="bar"><div class="bar-fill" style="width:{{ (rate * 100) | round }}%; background:{{ color }}"></div></div>
            <span class="pct" style="color:{{ color }}">{{ "%.1f" | format(rate * 100) }}%</span>
          </div>
        </td>
        <td><span class="badge {{ badge_class }}">{{ status }}</span></td>
        <td style="font-family:JetBrains Mono; color:var(--muted)">{{ "%.2f" | format(result.execution_time_s) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>

<!-- Distortion Stats -->
<div class="section">
  <div class="section-title">Distortion Statistics</div>
  <table>
    <thead><tr><th>Metric</th><th>Value</th><th>Limit</th><th>Status</th></tr></thead>
    <tbody>
      <tr>
        <td>Mean MSE Distortion</td>
        <td style="font-family:JetBrains Mono;color:var(--cyan)">{{ "%.6f" | format(distortion.mean_mse) }}</td>
        <td style="font-family:JetBrains Mono">0.040000</td>
        <td><span class="badge {{ 'badge-green' if distortion.mean_mse < 0.04 else 'badge-red' }}">
          {{ '✅ PASS' if distortion.mean_mse < 0.04 else '❌ FAIL' }}</span></td>
      </tr>
      <tr>
        <td>P95 MSE Distortion</td>
        <td style="font-family:JetBrains Mono;color:var(--cyan)">{{ "%.6f" | format(distortion.p95_mse) }}</td>
        <td style="font-family:JetBrains Mono">0.040000</td>
        <td><span class="badge {{ 'badge-green' if distortion.p95_mse < 0.04 else 'badge-red' }}">
          {{ '✅ PASS' if distortion.p95_mse < 0.04 else '❌ FAIL' }}</span></td>
      </tr>
      <tr>
        <td>P95 Latency</td>
        <td style="font-family:JetBrains Mono;color:var(--violet)">{{ "%.0f" | format(latency.p95) }}ms</td>
        <td style="font-family:JetBrains Mono">2000ms</td>
        <td><span class="badge {{ 'badge-green' if latency.p95 < 2000 else 'badge-red' }}">
          {{ '✅ PASS' if latency.p95 < 2000 else '❌ FAIL' }}</span></td>
      </tr>
    </tbody>
  </table>
</div>

<footer>APE-BAIT v1.0.0 · Research Build · {{ timestamp }}</footer>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="APE-BAIT Report Generator")
    parser.add_argument("--input", required=True, help="JSON validation report path")
    parser.add_argument("--output", default=None, help="Output HTML path")
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    with open(input_path) as f:
        report = json.load(f)
    
    # Ensure required keys with defaults
    report.setdefault("summary", {
        "aggregate_evasion_rate": 0, "avg_mse": 0, "avg_latency_ms": 0,
        "tools_tested": 0, "tools_evaded": 0, "mse_below_threshold": False,
        "p95_latency_ms": 0,
    })
    report.setdefault("evasion_by_tool", {})
    report.setdefault("distortion", {"mean_mse": 0, "p95_mse": 0, "max_mse": 0})
    report.setdefault("latency", {"p95": 0, "mean": 0})
    report.setdefault("metadata", {"method": "unknown", "epsilon": 0.03,
                                    "input_pcap": "N/A", "perturbed_pcap": "N/A"})
    
    # Render template
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        **report,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    )
    
    output_path = args.output or str(input_path.with_suffix(".html"))
    with open(output_path, "w") as f:
        f.write(html)
    
    print(f"✅ Report generated: {output_path}")


if __name__ == "__main__":
    main()
