# EURO_AI Setup & Troubleshooting Guide

Comprehensive guide for setting up the EURO_AI project, configuring environments, and resolving common issues.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Virtual Environment Setup](#virtual-environment-setup)
3. [Installing Dependencies](#installing-dependencies)
4. [GPU/CUDA Setup (Optional)](#gpucuda-setup-optional)
5. [MetaTrader5 Configuration](#metatrader5-configuration)
6. [Trained Models Setup](#trained-models-setup)
7. [Running the Server](#running-the-server)
8. [Troubleshooting](#troubleshooting)
9. [Development Workflow](#development-workflow)

---

## System Requirements

- **OS:** Windows 10 / 11 (for MetaTrader5 support)
  - macOS: fully supported for model development and inference, but note that the `MetaTrader5` package is Windows-only; use the MT5 terminal or remote Windows host for live trading.
- **Python:** 3.11 (recommended; 3.10 also works; **avoid 3.14+**)
- **RAM:** 8GB minimum (16GB+ recommended for model inference)
- **GPU (optional):** NVIDIA GPU with CUDA compute capability 3.5+
- **Disk Space:** 5GB+ for venv + models

### Check Python Installation

```powershell
# List available Python versions
py -0p

# Verify Python 3.11 is installed
py -3.11 --version
```

If Python 3.11 is not installed:
- Download from https://www.python.org/downloads/
- Install and ensure "Add Python to PATH" is checked

### macOS: use `pyenv` to manage Python versions (recommended)

On macOS it's convenient to install and manage Python versions with `pyenv` so you can create a Python 3.11 environment even when the system Python is newer.

```bash
# Install prerequisites (Homebrew)
brew update
brew install openssl readline zlib xz sqlite3 pyenv

# Add pyenv init to zsh
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
source ~/.zshrc

# Install Python 3.11 and create a venv
pyenv install 3.11.4          # or latest 3.11.x
pyenv local 3.11.4
python -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r lib/requirements.txt
```

If you prefer not to use `pyenv`, install a 3.11 distribution from python.org and create the venv with that interpreter.

---

## Virtual Environment Setup

### Create a New Virtual Environment

```powershell
# Create venv using Python 3.11
py -3.11 -m venv .venv311

# Activate it (PowerShell)
.\.venv311\Scripts\Activate.ps1

# Or in CMD
.\.venv311\Scripts\activate.bat
```

### Verify Activation

```powershell
# Should show path ending in .venv311\Scripts
python -c "import sys; print(sys.executable)"
```

---

## Installing Dependencies

### Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### Install from requirements.txt

```powershell
pip install -r requirements.txt
```

### Check Installation

```powershell
python -c "import fastapi, torch, pandas, joblib; print('All dependencies OK')"
```

---

## GPU/CUDA Setup (Optional)

### Check GPU Availability

```powershell
# Verify NVIDIA GPU is present
nvidia-smi
```

**Expected output:**
- GPU name (e.g., NVIDIA GeForce RTX 2050)
- Driver version
- CUDA version supported by driver

### Install PyTorch with CUDA

**Option 1: CUDA 12.1 (Recommended)**

```powershell
.\.venv311\Scripts\Activate.ps1

# Remove CPU-only PyTorch first
pip uninstall torch torchvision torchaudio -y

# Install GPU version
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio --upgrade
```

**Option 2: CUDA 11.8**

```powershell
pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch torchvision torchaudio --upgrade
```

### Verify CUDA Installation

```powershell
python -c "import torch; \
    print('Version:', torch.__version__); \
    print('CUDA:', torch.version.cuda); \
    print('Available:', torch.cuda.is_available()); \
    print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

**Expected output (if successful):**
```
Version: 2.9.1+cu121
CUDA: 12.1
Available: True
Device: NVIDIA GeForce RTX 2050
```

### Troubleshooting CUDA

**Problem:** `torch.cuda.is_available()` returns False

**Solutions:**
1. Check driver version: `nvidia-smi` should show "NVIDIA-SMI X.XX" and "CUDA Version X.X"
2. Reinstall latest driver from https://www.nvidia.com/Download/driverDetails.aspx
3. Ensure CUDA toolkit version in PyTorch wheel is <= driver CUDA version
   - Driver CUDA 13.1 → compatible with cu121, cu118, cu117
   - Driver CUDA 12.1 → compatible with cu121, cu118, cu117

---

## MetaTrader5 Configuration

### Option 1: Using MT5 Terminal (Recommended)

If you have MetaTrader5 installed and logged in:

```powershell
.\.venv311\Scripts\Activate.ps1

# Test connection
python -c "import MetaTrader5 as mt5; mt5.initialize(); print('Connected to MT5')"
```

### Option 2: Environment Variable Login

Edit `.env` with your MT5 credentials:

```env
MT5_LOGIN=1234567
MT5_PASSWORD=your_password
MT5_SERVER=YourBrokerServer
```

Then run the server — it will auto-authenticate.

### Getting Your MT5 Server Name

1. Open MetaTrader5 terminal
2. File → Open an Account (or Account settings)
3. Look for "Server:" field (e.g., "XM.COM-DEMO", "FXCM-Demo")

---

## Trained Models Setup

### Directory Structure

Create the following structure (if not already present):

```
prepared_datasets/
└── boosting_dl_residual/
    ├── eurusd_struct_meta.json
    └── trained_models/
        ├── lgb_gap_next.pkl
        ├── lgb_range_next.pkl
        ├── lgb_body_next.pkl
        ├── tcn_residual.pth
        └── dl_scaler.pkl
```

### eurusd_struct_meta.json Format

```json
{
  "feature_cols": [
    "Close", "ret_1", "ret_4", "ret_12",
    "ema_20", "ema_50", "ema_100",
    "rsi_14", "atr_14", "vol_20",
    "candle_body", "candle_range",
    "upper_wick", "lower_wick",
    "hour", "dayofweek",
    "session_asia", "session_london", "session_ny",
    "Spread"
  ],
  "seq_len": 24,
  "horizon": 1,
  "targets_boosting": ["gap_next", "range_next", "body_next"]
}
```

### Using Alternate Model Path

Set environment variable:

```powershell
$env:EURO_AI_BASE_DIR='C:\Full\Path\To\boosting_dl_residual'
```

Then restart the server.

---

## Running the Server

### Basic Start

```powershell
.\.venv311\Scripts\Activate.ps1
uvicorn server:app --host 0.0.0.0 --port 8000
```

Server will be available at `http://0.0.0.0:8000`

### VS Code: point the Python interpreter to the created venv

If you use VS Code and Pylance shows "Import ... could not be resolved", make sure VS Code is using the project's virtual environment:

1. Open the Command Palette: `Cmd+Shift+P` → `Python: Select Interpreter`.
2. Choose the interpreter at `<workspace>/.venv311/bin/python`.

Alternatively add the workspace override in `.vscode/settings.json` (created automatically by this repo):

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv311/bin/python",
  "python.analysis.indexing": true
}
```

### Development Mode (with auto-reload)

```powershell
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Custom Port

```powershell
uvicorn server:app --host 0.0.0.0 --port 9000
```

### Check Health

```powershell
# In another terminal
curl http://127.0.0.1:8000/health

# Or PowerShell
Invoke-RestMethod http://127.0.0.1:8000/health
```

---

## Troubleshooting

### Issue 1: ModuleNotFoundError: No module named 'MetaTrader5'

**Cause:** Using Python 3.14+; MetaTrader5 wheel not available for that version.

**Solution:**
```powershell
# Verify Python version
python --version

# If 3.14, use .venv311 instead:
.\.venv311\Scripts\Activate.ps1
python --version  # Should show 3.11.x
```

### Issue 2: MT5 Connection Failed: "Authorization failed"

**Cause:** MT5 terminal not running or not logged in.

**Solution (pick one):**
- Open MetaTrader5 terminal and log in manually
- Set env vars in `.env`:
  ```env
  MT5_LOGIN=your_account
  MT5_PASSWORD=your_password
  MT5_SERVER=YourBrokerServer
  ```

**To find broker server name:**
1. Open MT5 terminal
2. File → Open an Account
3. Look for "Server:" dropdown
4. Copy the server name (e.g., "XM.COM-DEMO")

### Issue 3: FileNotFoundError: 'lgb_gap_next.pkl'

**Cause:** Trained model files missing from expected location.

**Solution:**
```powershell
# Option 1: Copy models to default location
# Copy your models to: prepared_datasets\boosting_dl_residual\trained_models\

# Option 2: Set custom path
$env:EURO_AI_BASE_DIR='C:\path\to\your\boosting_dl_residual'

# Restart server
```

### Issue 4: CUDA not detected (but GPU present)

**Cause:** GPU drivers outdated or PyTorch CPU-only build.

**Solution:**
```powershell
# Check driver
nvidia-smi

# If outdated, update from https://www.nvidia.com/Download/

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio -y
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio --upgrade

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue 5: Port 8000 Already in Use

**Cause:** Another process using port 8000.

**Solution:**
```powershell
# Use different port
uvicorn server:app --port 9000

# Or find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue 6: Slow Inference (on CPU)

**Cause:** Running on CPU instead of GPU.

**Solution:**
- Install CUDA-enabled PyTorch (see [GPU/CUDA Setup](#gpucuda-setup-optional))
- Check `/health` endpoint — `device` should show `cuda`
- Expected speedup: 3-10x on NVIDIA GPU

### Issue 7: ImportError in server.py

**Cause:** Missing module imports in virtual environment.

**Solution:**
```powershell
# Ensure venv is activated
.\.venv311\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt

# Verify specific import
python -c "from mt5_utils import mt5_connect; print('OK')"
```

---

## Development Workflow

### Code Changes

1. Make edits to `server.py`, `mt5_utils.py`, etc.
2. If running with `--reload`, changes auto-apply
3. Otherwise, restart the server (Ctrl+C, then re-run)

### Adding New Dependencies

```powershell
.\.venv311\Scripts\Activate.ps1

# Install new package
pip install new_package_name

# Update requirements.txt
pip freeze > requirements.txt.new
# Review and merge changes manually or use:
pip install pipdeptree  # to visualize dependencies
```

### Testing Endpoints

**Using PowerShell:**

```powershell
# GET /health
Invoke-RestMethod http://127.0.0.1:8000/health

# POST /predict_mt5
$body = @{
    symbol = 'EURUSD'
    timeframe = 'H1'
    bars = 300
} | ConvertTo-Json

Invoke-RestMethod -Uri http://127.0.0.1:8000/predict_mt5 `
    -Method POST `
    -Body $body `
    -ContentType 'application/json'

# GET /logs
Invoke-RestMethod http://127.0.0.1:8000/logs
```

**Using curl (if available):**

```bash
# GET
curl http://127.0.0.1:8000/health

# POST
curl -X POST http://127.0.0.1:8000/predict_mt5 \
  -H "Content-Type: application/json" \
  -d '{"symbol":"EURUSD","timeframe":"H1","bars":300}'
```

### Debugging

Enable verbose logging (edit `server.py`):

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Then restart the server.

---

## File Structure Reference

```
d:\Project\AI Model\EURO_AI\
├── .env                              # Your env vars (git-ignored)
├── .env.example                      # Template for .env
├── .gitignore                        # Git exclusion rules
├── .venv311/                         # Virtual environment (git-ignored)
│   ├── Scripts/
│   │   ├── Activate.ps1
│   │   ├── python.exe
│   │   └── pip.exe
│   └── Lib/site-packages/            # Installed packages
├── README.md                         # Quick start guide
├── SETUP.md                          # This file
├── requirements.txt                  # Dependencies
├── server.py                         # FastAPI app (main)
├── mt5_utils.py                      # MT5 utilities (separated)
├── mt5.py                            # Legacy MT5 module
├── data_csv/                         # Historical OHLC data
│   ├── EURUSD_D1.csv
│   ├── EURUSD_H1.csv
│   └── EURUSD_H4.csv
├── models/                           # Model configs/metadata
├── prepared_datasets/
│   └── boosting_dl_residual/
│       ├── eurusd_struct_meta.json
│       └── trained_models/           # Add your models here
│           ├── lgb_gap_next.pkl
│           ├── lgb_range_next.pkl
│           ├── lgb_body_next.pkl
│           ├── tcn_residual.pth
│           └── dl_scaler.pkl
└── jupyter/                          # (Optional) Jupyter notebooks

```

---

## Quick Commands Reference

```powershell
# Activate venv
.\.venv311\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Check Python version
python --version

# Check GPU availability
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Run server (production)
uvicorn server:app --host 0.0.0.0 --port 8000

# Run server (development with reload)
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Test health endpoint
Invoke-RestMethod http://127.0.0.1:8000/health
```

---

## Further Help

- **FastAPI Docs:** https://fastapi.tiangolo.com
- **MetaTrader5 Python:** https://www.mql5.com/en/docs/integration/python_metatrader5
- **PyTorch CUDA:** https://pytorch.org/get-started/locally/
- **LightGBM:** https://lightgbm.readthedocs.io

For project-specific issues, check the README.md or open an issue on GitHub.

---

**Last Updated:** December 2025
