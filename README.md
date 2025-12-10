# EURO_AI — EUR/USD Price Prediction with ML + Deep Learning

A FastAPI-based server that combines **LightGBM** (boosting) and **TCN** (temporal convolutional networks) models to forecast EUR/USD OHLC candles using MetaTrader5 (MT5) live data. The system predicts gaps, ranges, and candle bodies for the next trading period.

---

## Features

- **Dual-model prediction**: LightGBM for base predictions + TCN residuals for refinement
- **FastAPI REST API**: Real-time predictions via HTTP endpoints
- **MetaTrader5 integration**: Auto-fetch live rates from MT5 terminal
- **Graceful fallbacks**: Models/MT5 connection failures don't crash the server
- **Feature engineering**: 20+ technical indicators (RSI, ATR, EMA, candle patterns, session info)
- **Modular architecture**: Separated MT5 utilities (`mt5_utils.py`) from the API

---

## Quick Start

### 1. Create Virtual Environment

```bash
# Create venv with Python 3.11
py -3.11 -m venv .venv311

# Activate
.\.venv311\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. (Optional) Install CUDA Support

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio --upgrade
```

### 3. Configure Environment

Copy and edit `.env`:

```bash
cp .env.example .env
# Edit .env with your MT5 credentials (or leave empty if terminal is logged in)
```

### 4. Run Server

```bash
.\.venv311\Scripts\Activate.ps1
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at `http://0.0.0.0:8000`

---

## API Endpoints

### GET `/health`
Check server status.

### POST `/predict_mt5`
Fetch live rates from MT5 and predict next candle.

```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "bars": 300
}
```

### GET `/logs`
Retrieve request history (last 1000 entries).

---

## Project Structure

```
EURO_AI/
├── server.py                  # FastAPI app + endpoints
├── mt5_utils.py               # MT5 utilities (isolated)
├── mt5.py                     # Legacy MT5 module
├── requirements.txt           # Dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git exclusion rules
├── SETUP.md                   # Detailed setup guide
├── README.md                  # This file
├── data_csv/                  # Historical OHLC data
├── models/                    # Model configs
├── prepared_datasets/
│   └── boosting_dl_residual/
│       ├── eurusd_struct_meta.json
│       └── trained_models/
│           ├── lgb_gap_next.pkl
│           ├── lgb_range_next.pkl
│           ├── lgb_body_next.pkl
│           ├── tcn_residual.pth
│           └── dl_scaler.pkl
└── .venv311/                  # Python 3.11 environment
```

---

## Troubleshooting

### MetaTrader5 Import Error

- Use **Python 3.11** (not 3.14)
- MetaTrader5 wheel unavailable for Python 3.14 on PyPI
- Use `.venv311` created with `py -3.11`

### MT5 Connection Error

- **Option 1:** Open MT5 terminal and log in (recommended)
- **Option 2:** Set env vars:
  ```bash
  $env:MT5_LOGIN='your_login'
  $env:MT5_PASSWORD='your_password'
  $env:MT5_SERVER='BrokerServer'
  ```

### Missing Model Files

- Set `EURO_AI_BASE_DIR` to path containing `trained_models/`:
  ```bash
  $env:EURO_AI_BASE_DIR='C:\path\to\boosting_dl_residual'
  ```
- Or place models in `prepared_datasets/boosting_dl_residual/trained_models/`

### CUDA Not Working

- Check `nvidia-smi` for GPU drivers
- Reinstall PyTorch with CUDA:
  ```bash
  pip uninstall torch -y
  pip install --index-url https://download.pytorch.org/whl/cu121 torch --upgrade
  ```
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

---

## Environment Variables

| Var | Description |
|-----|-------------|
| `EURO_AI_BASE_DIR` | Path to `boosting_dl_residual` folder |
| `MT5_LOGIN` | MT5 account number |
| `MT5_PASSWORD` | MT5 password |
| `MT5_SERVER` | MT5 broker server |

---

## Development

- **Hot-reload:** `uvicorn server:app --reload`
- **Code structure:** See SETUP.md for details
- **GPU acceleration:** See SETUP.md for CUDA installation

---

**See `SETUP.md` for comprehensive setup and troubleshooting guide.**
