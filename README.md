# Monthly UK Borrowing Forecaster

## Overview

This project is a Python-based forecasting tool for **UK Public Sector Net Borrowing (PSNB)** using official monthly ONS time-series data. It provides two ways to use the project:

1. **Batch / command-line workflow** using `main.py`
2. **Interactive Streamlit GUI** using `app.py`

The tool validates the input series, generates 1-to-12 month forecasts, runs expanding-window backtests, calculates forecast accuracy and interval coverage metrics, and exports outputs such as CSV, JSON, and PNG files.

> **Important:** This project is for educational, research, and analytical purposes only. It is **not financial advice**.

---

## What the project does

The project was designed as an open analytical tool for monthly UK borrowing forecasts, with reproducible outputs and clear provenance. The current active forecasting methods in the codebase are:

- **Seasonal naïve**
- **ETS (Error, Trend, Seasonal exponential smoothing)**

ARIMA remains in the codebase as part of project exploration, but it is not currently enabled in the final active scope.

---

## Project structure

### Product package contents

This submission includes all files needed to run the product locally from the submitted folder:

- source code files
- setup scripts
- dependency list
- default dataset
- this README file
- an `outputs/` folder location for generated results

### Core application files

- **`main.py`**  
  Batch / command-line entry point. Runs the full pipeline:
  - load the dataset
  - validate monthly structure
  - generate forecasts
  - run expanding-window backtests
  - compute error and coverage metrics
  - export results into a timestamped folder inside `outputs/`

- **`app.py`**  
  Streamlit user interface. Supports:
  - forecast generation
  - historical evaluation
  - single-method analysis
  - comparison of active methods
  - chart rendering
  - CSV export from the GUI

### Data processing and modelling

- **`ons_io.py`**  
  Loads ONS CSV or Excel time-series files and converts them into a monthly pandas Series with metadata.

- **`validate.py`**  
  Validates the time series before modelling. Checks for problems such as:
  - non-datetime index
  - duplicate timestamps
  - unsorted dates
  - missing values
  - missing months
  - incorrect monthly anchoring

- **`models.py`**  
  Contains the forecasting models used by the project. Active methods are:
  - seasonal naïve
  - ETS

- **`backtest.py`**  
  Runs the expanding-window backtest process and compares forecasts against realised values.

- **`metrics.py`**  
  Computes evaluation metrics such as:
  - MAE
  - RMSE
  - 80% interval coverage
  - 95% interval coverage
  - average interval widths

- **`plot.py`**  
  Creates and saves forecast charts as PNG files, including forecast paths and prediction intervals.

### Setup and dependencies

- **`requirements.txt`**  
  Pinned Python dependencies for the project.

- **`setup.ps1`**  
  Windows PowerShell setup script. Creates a virtual environment and installs dependencies.

- **`setup.sh`**  
  macOS / Linux setup script. Creates a virtual environment and installs dependencies.

### Data

- **`series-050326.csv`**  
  Default sample dataset used by both the batch workflow and the Streamlit app.

### Folder structure summary

The product should be run from the project root folder containing:

- `main.py`
- `app.py`
- `backtest.py`
- `metrics.py`
- `models.py`
- `ons_io.py`
- `plot.py`
- `validate.py`
- `requirements.txt`
- `setup.ps1`
- `setup.sh`
- `series-050326.csv`
- `README.md`

---

## Requirements

- Python 3.11+ is recommended
- A working terminal / shell
- Internet access for installing dependencies

---

## Setup

## Windows setup (PowerShell)

From the project folder, run:

```powershell
./setup.ps1
```

If PowerShell blocks script execution, you may need to allow local scripts for the current session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
./setup.ps1
```

The script will:

1. Create a virtual environment called `.venv`
2. Activate it
3. Upgrade `pip`
4. Install packages from `requirements.txt`

### Manual Windows setup

If you prefer to run the commands yourself:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

---

## macOS / Linux setup

From the project folder, run:

```bash
bash setup.sh
```

The script will:

1. Create a virtual environment called `.venv`
2. Activate it
3. Upgrade `pip`
4. Install packages from `requirements.txt`

### Manual macOS / Linux setup

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to run the project

There are **two versions** of the project you can run.

## 1) Run the batch / CLI version

This version runs the full end-to-end pipeline and exports files to a timestamped output folder.

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

### macOS / Linux

```bash
source ./.venv/bin/activate
python3 main.py
```

### What this produces

Running `main.py` creates a new timestamped folder inside `outputs/` containing files such as:

- `forecast.csv`
- `backtest_predictions.csv`
- `errors.csv`
- `coverage.csv`
- `error_summary.csv`
- `coverage_summary.csv`
- `validation_report.json`
- `run_summary.json`
- `forecast_seasonal_naive.png`
- `forecast_ets.png`

It also prints a short run summary in the terminal when the pipeline completes successfully.

---

## 2) Run the Streamlit GUI version

This version launches an interactive web app in your browser.

**Important:** Run this command from the **project root folder** — the folder that contains `app.py`. If you run it from another directory, Streamlit will not find the file correctly.

### Windows

```powershell
cd path\to\your\project-folder
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

### macOS / Linux

```bash
cd /path/to/your/project-folder
source ./.venv/bin/activate
streamlit run app.py
```

### Accessing the Streamlit app

After running `streamlit run app.py`, Streamlit will print a local URL in the terminal. Open that link in your browser to use the app. In a standard local run this is usually:

```text
http://localhost:8501
```

No separate hosted deployment is included in this submission. The web-based part of the project is intended to be run locally from the submitted package.

### What you can do in the GUI

The Streamlit app supports:

- **Forecast generation**
- **Historical evaluation**
- selecting the **forecast method**
- choosing a **forecast horizon** from 1 to 12 months
- choosing how much **history** to show on charts
- comparing all active methods during historical evaluation
- downloading CSV output from the interface

The app uses `series-050326.csv` as the default dataset.

---

## Instructions for markers to install and test the product

### Install

Follow either the **Windows setup** or **macOS / Linux setup** steps above.

### Test the batch / CLI version

Run:

```powershell
python main.py
```

or on macOS / Linux:

```bash
python3 main.py
```

Expected result:

- a new timestamped folder is created inside `outputs/`
- forecast, backtest, metrics, validation, and PNG files are generated
- the terminal prints a short run summary

### Test the Streamlit GUI version

Run:

```powershell
streamlit run app.py
```

or on macOS / Linux:

```bash
streamlit run app.py
```

Expected result:

- Streamlit starts successfully
- a local URL appears in the terminal
- the app opens in a browser
- forecasts and historical evaluation can be run from the GUI
- CSV outputs can be downloaded from the interface

---

## Expected workflow

### Batch / CLI workflow

Use `main.py` when you want:

- reproducible full pipeline runs
- exported forecast and evaluation artefacts
- timestamped output folders
- a non-interactive workflow suitable for submission or testing

### Streamlit GUI workflow

Use `app.py` when you want:

- an interactive interface
- quick exploration of forecasts
- historical evaluation from chosen origin months
- easy chart viewing and CSV downloads

---

## Notes

- The current implementation expects the dataset file to be available in the project folder.
- Both `main.py` and `app.py` use `series-050326.csv` as the default input dataset.
- The project is intended for transparency, reproducibility, and academic analysis.
- Forecast outputs should be interpreted carefully and in context.

> **Disclaimer:** This software and its outputs are provided for educational and analytical purposes only. They do **not** constitute financial advice, investment advice, or policy advice.

---

## Quick start

### Windows

```powershell
./setup.ps1
python main.py
streamlit run app.py
```

### macOS / Linux

```bash
bash setup.sh
python3 main.py
streamlit run app.py
```
