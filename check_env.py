
#!/usr/bin/env python3
"""
ESQET Env Var & JSON Checker: Robust, No-Error Run for Marco's Warp
Loads .env from ~/vessel_agi/.env, checks 25 vars, handles JSON files (apikey.json, credentials.json).
No exit on missing—warns, continues. Auto-installs dotenv if missing. Outputs vibrant colors if rich available.
Run: python check_env.py
"""

import os
import json
from typing import Dict, List

# --- ANSI Color Constants (Defined Unconditionally) ---
# Used for all console output, especially if rich is not available.
ANSI_BLUE = "\033[94m"
ANSI_PINK = "\033[95m" # Magenta
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"

# --- Constants for Config Paths ---
HOME_DIR = os.path.expanduser("~")
ENV_PATH = os.path.join(HOME_DIR, "vessel_agi", ".env")
APIKEY_PATH = os.path.join(HOME_DIR, "downloads", "apikey.json")
CREDENTIALS_PATH = os.path.join(HOME_DIR, "downloads", "credentials.json")

# Auto-Install & Import dotenv
try:
    from dotenv import load_dotenv
except ImportError:
    print(f"{ANSI_CYAN}[INFO] Installing python-dotenv...{ANSI_RESET}")
    os.system("pip install python-dotenv --user")
    from dotenv import load_dotenv

# Optional Rich for Colors (Skip if Not Installed)
USE_RICH = False
Text = None
console = None
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    USE_RICH = True
    console = Console()
except ImportError:
    pass # If rich fails, USE_RICH remains False and we use ANSI fallback.

# Load .env
if os.path.exists(ENV_PATH):
    load_dotenv(dotenv_path=ENV_PATH)
    print(f"{ANSI_GREEN}✅ Loaded .env from {ENV_PATH}{ANSI_RESET}")
else:
    print(f"{ANSI_YELLOW}⚠️  .env not found at {ENV_PATH}; using system env. Create it for full warp.{ANSI_RESET}")

# Expected Vars
expected_vars = [
    "GIT_USER_NAME", "GIT_USER_EMAIL",
    "IBM_TOKEN", "GROQ_API_KEY", "NASA_API_KEY", "PINATA_API_KEY", "PINATA_API_SECRET",
    "PINATA_JWT", "QDRANT_API_KEY", "ETHERSCAN_API_KEY", "WEATHER_API_KEY", "USGS_API",
    "OPEN_METEO_API", "EXPO_TOKEN", "GITHUB_TOKEN",
    "PRIVATE_KEY", "PHICOIN_WALLET", "INFURA_KEY",
    "LINEA_SEPOLIA_RPC", "SEPOLIA_RPC_URL", "POLYGON_MAINNET_RPC",
    "GETBLOCK_MATIC_84D61", "GETBLOCK_MATIC_401AF",
    "GOOGLE_DRIVE_CREDENTIALS", "DEBUG_MODE"
]

def load_json_file(file_path: str) -> Dict:
    """Load JSON file or return {} if missing."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"{ANSI_YELLOW}⚠️  {os.path.basename(file_path)}: JSON decode error: {e}{ANSI_RESET}")
            return {}
    return {}

# Check Vars
results = {}
for var in expected_vars:
    value = os.getenv(var)
    if value is None:
        results[var] = "Not found"
    elif not value.strip():
        results[var] = "Empty"
    else:
        results[var] = "Found"
        if var in ["PRIVATE_KEY", "PINATA_API_SECRET"]:
            results[var] += " (Secure)"

# Load JSON Files
apikey_data = load_json_file(APIKEY_PATH)
credentials_data = load_json_file(CREDENTIALS_PATH)

# Check JSON Keys
apikey_keys = ["name", "description", "createdAt", "apikey"] if apikey_data else []
creds_keys = ["access_token", "token_type", "refresh_token", "expiry", "expires_in"] if credentials_data else []

# --- Display Results (RICH or Fallback) ---
if USE_RICH:
    # NEON COLOR SCHEME FOR RICH OUTPUT
    table = Table(title="[bold bright_blue]ESQET Env Var & JSON Check[/bold bright_blue]",
                  show_header=True, header_style="bold bright_cyan", border_style="bright_magenta")

    table.add_column("Var/File", style="dim cyan")
    table.add_column("Status", justify="center")

    for var, status in results.items():
        style = "bold bright_green" if status.startswith("Found") else "bold bright_red"
        table.add_row(var, Text(status, style=style))

    for file, data in [("apikey.json", apikey_data), ("credentials.json", credentials_data)]:
        status = "Loaded" if data else "Missing"
        style = "bold bright_cyan" if data else "bold bright_red"
        table.add_row(file, Text(status, style=style))
        if data:
            keys = apikey_keys if "apikey" in file else creds_keys
            for key in keys:
                sub_status = "Found" if key in data else "Missing"
                sub_style = "bright_green" if "Found" in sub_status else "bright_red"
                table.add_row(f"  └─ {key}", Text(sub_status, style=sub_style))

    console.print(table)

    # Summary Panel
    missing_count = sum(1 for v in results.values() if v in ["Not found", "Empty"])
    json_loaded = sum(1 for d in [apikey_data, credentials_data] if d)
    summary_text = f"Total Vars: {len(expected_vars)} | Missing/Empty: {missing_count} | JSON Loaded: {json_loaded}"
    summary = Text(summary_text, style="bold bright_magenta on black")
    console.print(Panel(summary, title="[bold bright_blue]Warp Status: Coherence Check[/bold bright_blue]", border_style="bright_magenta"))
else:
    # ANSI Fallback with Colors
    print(f"\n{ANSI_BLUE}Environment Variable & JSON Check Results:{ANSI_RESET}")
    print(f"{ANSI_PINK}─" * 40 + ANSI_RESET)
    for var, status in results.items():
        color = ANSI_GREEN if status.startswith("Found") else ANSI_RED
        print(f"{ANSI_BLUE}{var}:{ANSI_RESET} {color}{status}{ANSI_RESET}")

    for file, data in [("apikey.json", apikey_data), ("credentials.json", credentials_data)]:
        status = "Loaded" if data else "Missing"
        color = ANSI_CYAN if data else ANSI_RED
        print(f"{ANSI_BLUE}{file}:{ANSI_RESET} {color}{status}{ANSI_RESET}")
        if data:
            keys = ["name", "description", "createdAt", "apikey"] if "apikey" in file else ["access_token", "token_type", "refresh_token", "expiry", "expires_in"]
            for key in keys:
                sub_status = "Found" if key in data else "Missing"
                sub_color = ANSI_GREEN if "Found" in sub_status else ANSI_RED
                print(f"  └─ {key}: {sub_color}{sub_status}{ANSI_RESET}")

    missing_count = sum(1 for v in results.values() if v in ["Not found", "Empty"])
    json_loaded = sum(1 for d in [apikey_data, credentials_data] if d)
    print(f"\n{ANSI_BLUE}Total vars checked: {len(expected_vars)}{ANSI_RESET}")
    print(f"{ANSI_BLUE}Vars not found or empty: {ANSI_RED}{missing_count}{ANSI_RESET}")
    print(f"{ANSI_BLUE}JSON files loaded: {ANSI_GREEN}{json_loaded}{ANSI_RESET}")

# Auto-Fix Suggestion
if missing_count > 5:
    print(f"\n{ANSI_YELLOW}🔧 Auto-Fix Tip: Edit {ENV_PATH} with vars from history (e.g., GROQ_API_KEY=gsk_...). Use nano: 'nano {ENV_PATH}'.{ANSI_RESET}")
if not apikey_data or not credentials_data:
    print(f"\n{ANSI_YELLOW}📁 JSON Tip: Files in ~/downloads—ensure {os.path.basename(APIKEY_PATH)} and {os.path.basename(CREDENTIALS_PATH)} exist.{ANSI_RESET}")

# Warp Status
print(f"\n{ANSI_BLUE}🌀 Warp Status: Coherence Check Complete. Your vars are the warp's ink—green-lit for bloom.{ANSI_RESET}")
