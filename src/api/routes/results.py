"""Results and statistics routes."""
from fastapi import APIRouter, HTTPException
from src.api.models import ToolInfo, TargetTool

router = APIRouter()

TOOL_CATALOG: dict[TargetTool, dict] = {
    TargetTool.SNORT:              {"name": "Snort",                     "category": "Signature IDS",      "description": "Open-source network IDS using rule-based detection.",                    "typical_evasion_rate": "60-70%"},
    TargetTool.SURICATA:           {"name": "Suricata",                  "category": "Signature IDS",      "description": "Multi-threaded IDS/IPS with protocol analysis.",                        "typical_evasion_rate": "55-65%"},
    TargetTool.YARA:               {"name": "YARA",                      "category": "Static Analysis",    "description": "Pattern-matching tool for malware identification.",                      "typical_evasion_rate": "20-30%"},
    TargetTool.CLAMAV:             {"name": "ClamAV",                    "category": "Signature AV",       "description": "Open-source antivirus with signature-based detection.",                  "typical_evasion_rate": "75-85%"},
    TargetTool.ZEEK:               {"name": "Zeek",                      "category": "Network Monitor",    "description": "Network analysis framework for behavioral detection.",                   "typical_evasion_rate": "78-88%"},
    TargetTool.MALCONV:            {"name": "MalConv",                   "category": "ML Malware",         "description": "End-to-end deep learning on raw bytes for malware detection.",           "typical_evasion_rate": "85-95%"},
    TargetTool.CNN_IDS:            {"name": "CNN IDS",                   "category": "ML IDS",            "description": "Convolutional neural network trained on network flow features.",          "typical_evasion_rate": "80-90%"},
    TargetTool.RANDOM_FOREST:      {"name": "Random Forest IDS",         "category": "ML IDS",            "description": "Ensemble decision tree classifier on tabular traffic features.",          "typical_evasion_rate": "82-90%"},
    TargetTool.LSTM_IDS:           {"name": "LSTM IDS",                  "category": "ML IDS",            "description": "Recurrent network modeling temporal traffic sequences.",                  "typical_evasion_rate": "78-88%"},
    TargetTool.ISOLATION_FOREST:   {"name": "Isolation Forest",          "category": "Anomaly Detection", "description": "Unsupervised anomaly detector using random partitioning.",                "typical_evasion_rate": "75-85%"},
    TargetTool.LOF:                {"name": "Local Outlier Factor",      "category": "Anomaly Detection", "description": "Density-based local anomaly detector.",                                  "typical_evasion_rate": "72-82%"},
    TargetTool.AUTOENCODER:        {"name": "Autoencoder",               "category": "Anomaly Detection", "description": "Neural autoencoder detecting anomalies from reconstruction error.",      "typical_evasion_rate": "78-88%"},
    TargetTool.EMBER_GBM:          {"name": "EMBER LightGBM",            "category": "ML Malware",         "description": "Gradient boosted tree trained on the EMBER malware feature set.",        "typical_evasion_rate": "76-86%"},
    TargetTool.DGA_DETECTOR:       {"name": "DGA Detector",              "category": "DNS Analysis",       "description": "ML classifier detecting domain generation algorithm activity.",           "typical_evasion_rate": "68-78%"},
    TargetTool.PORT_SCAN_DETECTOR: {"name": "Port Scan Detector",        "category": "Heuristic",          "description": "Statistical detector for port scanning behavior.",                       "typical_evasion_rate": "60-75%"},
}


@router.get("/tools", response_model=list[ToolInfo])
async def list_tools():
    """List all supported target security tools with metadata."""
    return [
        ToolInfo(id=tool_id, **info)
        for tool_id, info in TOOL_CATALOG.items()
    ]


@router.get("/tools/{tool_id}", response_model=ToolInfo)
async def get_tool(tool_id: TargetTool):
    """Get metadata for a specific target tool."""
    if tool_id not in TOOL_CATALOG:
        raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
    return ToolInfo(id=tool_id, **TOOL_CATALOG[tool_id])
