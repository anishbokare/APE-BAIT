#!/usr/bin/env python
"""
APE-BAIT Validation Suite Runner
Runs perturbed traffic through 15+ security tools and generates a report.

Usage:
    python scripts/run_validation.py --pcap data/sample_pcaps/sample.pcap
    python scripts/run_validation.py --pcap traffic.pcap --method pgd --output results/
    python scripts/run_validation.py --tools snort,suricata,yara --pcap traffic.pcap
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from pathlib import Path

import click
import numpy as np

from src.core.config import load_config
from src.core.engine import APEBAITEngine
from src.core.logger import get_logger
from src.validation.runner import ValidationRunner
from src.validation.metrics import MetricsCalculator


logger = get_logger("ape-bait.validate")


def parse_args():
    parser = argparse.ArgumentParser(description="APE-BAIT Validation Suite")
    parser.add_argument("--pcap", required=True, help="Input PCAP file path")
    parser.add_argument("--output", default="results/", help="Output directory")
    parser.add_argument("--method", choices=["fgsm", "pgd", "ensemble"], default="pgd")
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--tools", default=None,
                        help="Comma-separated list of tools (default: all)")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--no-parallel", dest="parallel", action="store_false")
    parser.add_argument("--report-format", choices=["json", "html"], default="json")
    return parser.parse_args()


def generate_perturbed_pcap(
    input_pcap: str,
    method: str,
    epsilon: float,
    config_path: str = None,
) -> str:
    """Run APE-BAIT engine to generate perturbed PCAP."""
    logger.info(f"Generating perturbed PCAP: method={method} epsilon={epsilon}")
    
    overrides = {
        "perturbation": {"method": method, "epsilon": epsilon}
    }
    cfg = load_config(config_path=config_path, overrides=overrides)
    engine = APEBAITEngine(cfg)
    
    result = engine.run_demo(input_pcap)
    perturbed_path = result["output_pcap"]
    metrics = result["metrics"]
    
    logger.info(
        "Perturbation complete",
        perturbed_pcap=perturbed_path,
        avg_mse=metrics.get("avg_mse"),
        avg_latency_ms=metrics.get("avg_latency_ms"),
    )
    
    return perturbed_path, metrics


def main():
    args = parse_args()
    
    input_pcap = Path(args.pcap)
    if not input_pcap.exists():
        print(f"❌ PCAP file not found: {args.pcap}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tools = args.tools.split(",") if args.tools else None
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║         APE-BAIT Validation Suite Runner             ║
╠══════════════════════════════════════════════════════╣
║  Input PCAP  : {str(input_pcap):<36} ║
║  Method      : {args.method:<36} ║
║  Epsilon     : {args.epsilon:<36} ║
║  Output      : {str(output_dir):<36} ║
╚══════════════════════════════════════════════════════╝
""")
    
    # Step 1: Generate perturbed PCAP
    print("⚡ Step 1: Generating adversarial perturbations...")
    try:
        perturbed_pcap, engine_metrics = generate_perturbed_pcap(
            str(input_pcap),
            method=args.method,
            epsilon=args.epsilon,
            config_path=args.config,
        )
        print(f"   ✅ Perturbed PCAP: {perturbed_pcap}")
        print(f"   📊 Avg MSE: {engine_metrics.get('avg_mse', 'N/A')}")
        print(f"   ⏱  Avg Latency: {engine_metrics.get('avg_latency_ms', 'N/A')} ms")
    except Exception as e:
        print(f"   ❌ Perturbation failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    
    # Step 2: Run validation tools
    print("\n🔍 Step 2: Running validation suite...")
    
    cfg = load_config(config_path=args.config)
    cfg.validation.parallel = args.parallel
    runner = ValidationRunner(cfg)
    
    try:
        report = runner.run(
            original_pcap=str(input_pcap),
            perturbed_pcap=perturbed_pcap,
            tools=tools,
        )
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
    
    # Merge engine metrics into report
    report["engine_metrics"] = engine_metrics
    report["metadata"] = {
        "input_pcap": str(input_pcap),
        "perturbed_pcap": perturbed_pcap,
        "method": args.method,
        "epsilon": args.epsilon,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    
    # Step 3: Save report
    print("\n📋 Step 3: Generating report...")
    
    if args.report_format == "json":
        report_path = output_dir / f"validation_report_{int(time.time())}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"   ✅ Report saved: {report_path}")
    
    # Step 4: Print summary
    summary = report.get("summary", {})
    tool_results = report.get("evasion_by_tool", {})
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║                  VALIDATION SUMMARY                  ║
╠══════════════════════════════════════════════════════╣
║  Aggregate Evasion Rate : {f"{summary.get('aggregate_evasion_rate', 0)*100:.1f}%":<28} ║
║  Tools Tested           : {str(summary.get('tools_tested', 0)):<28} ║
║  Tools Evaded (≥50%)    : {str(summary.get('tools_evaded', 0)):<28} ║
║  Avg MSE Distortion     : {str(summary.get('avg_mse', 'N/A')):<28} ║
║  MSE Below Threshold    : {"✅ YES" if summary.get('mse_below_threshold') else "❌ NO":<28} ║
║  P95 Latency            : {f"{summary.get('p95_latency_ms', 0):.0f} ms":<28} ║
║  Latency Below 2s       : {"✅ YES" if summary.get('latency_below_2s') else "❌ NO":<28} ║
╠══════════════════════════════════════════════════════╣
║  TOOL-BY-TOOL RESULTS                                ║
╠══════════════════════════════════════════════════════╣""")
    
    for tool_name, result in tool_results.items():
        rate = result.get('evasion_rate', 0)
        status = "✅" if rate >= 0.5 else "⚠️ "
        err = "ERROR" if result.get("error") else f"{rate*100:.1f}%"
        print(f"║  {status} {tool_name:<20} : {err:<27} ║")
    
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\n✅ Validation complete! Report: {output_dir}")


@click.command()
@click.option("--pcap", required=True, help="Input PCAP file")
@click.option("--output", default="results/")
@click.option("--method", default="pgd")
@click.option("--epsilon", default=0.03)
def cli(pcap, output, method, epsilon):
    """APE-BAIT validation suite CLI entry point."""
    sys.argv = ["run_validation.py", "--pcap", pcap, "--output", output,
                "--method", method, "--epsilon", str(epsilon)]
    main()


if __name__ == "__main__":
    main()
