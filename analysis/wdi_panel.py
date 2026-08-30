import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1) Point this to your World Bank "Custom" WDI download folder
#    (the folder that contains the CSVs after you unzip).
# ------------------------------------------------------------
DATA_DIR = Path("data/raw/worldbank_wdi_download/P_Data_Extract_From_World_Development_Indicators")  # <-- change this
WDI_FILE = DATA_DIR / "05cad901-327c-43d2-b612-290e5d731230_Data.csv"  # <-- change to your exact filename

# Safety check (helps catch path/filename issues early)
if not WDI_FILE.exists():
    raise FileNotFoundError(f"Could not find WDI file at: {WDI_FILE}")

# ------------------------------------------------------------
# 2) Load the WDI file
#    World Bank WDI CSVs are "wide": years are columns.
#    They also contain extra columns like:
#    Country Name, Country Code, Series Name, Series Code
# ------------------------------------------------------------
df_wide = pd.read_csv(WDI_FILE)

# Keep identifier columns
id_cols = ["Country Name", "Country Code", "Series Name", "Series Code"]
missing_cols = [c for c in id_cols if c not in df_wide.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns in WDI file: {missing_cols}")

# Robust year column detection
year_cols = []
for c in df_wide.columns:
    c_str = str(c).strip()
    if c_str[:4].isdigit():
        y = int(c_str[:4])
        if 1960 <= y <= 2025:
            year_cols.append(c)

if not year_cols:
    raise ValueError("No year columns detected. Check the WDI CSV format.")

df_wide = df_wide[id_cols + year_cols].copy()

# Melt wide → long
df_long = df_wide.melt(
    id_vars=id_cols,
    value_vars=year_cols,
    var_name="year",
    value_name="value",
)

# Clean types
df_long["year"] = (
    df_long["year"]
    .astype(str)
    .str.extract(r"(\d{4})")[0]
    .astype(int)
)

df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

# ------------------------------------------------------------
# 3) Filter to ONLY the indicators you care about
# ------------------------------------------------------------
INDICATORS = {
    "NY.GDP.PCAP.KD": "gdp_per_capita_const2015_usd",
    "EN.GHG.CO2.PC.CE.AR5": "co2_per_capita_tons",
    "SP.POP.TOTL": "population",
}

# Show which of your desired codes are present (use INDICATORS keys to avoid mismatch)
present = set(df_long["Series Code"].dropna().unique())
wanted = list(INDICATORS.keys())
print("Wanted codes present:", [c for c in wanted if c in present])

missing_codes = [c for c in wanted if c not in present]
if missing_codes:
    print("WARNING: Missing codes in this WDI download:", missing_codes)

# Show top series codes by number of rows (quick sanity check)
print(df_long["Series Code"].value_counts().head(20))

df_long = df_long[df_long["Series Code"].isin(INDICATORS.keys())].copy()
df_long["indicator"] = df_long["Series Code"].map(INDICATORS)

# ------------------------------------------------------------
# 4) Pivot so you have one row per country-year with columns:
#    gdp_per_capita_const2015_usd, co2_per_capita_tons, population
# ------------------------------------------------------------
panel = (
    df_long.pivot_table(
        index=["Country Name", "Country Code", "year"],
        columns="indicator",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)

# Optional: standardize column names for later code
panel = panel.rename(
    columns={
        "Country Name": "country",
        "Country Code": "iso3c",
    }
)

# ------------------------------------------------------------
# 5) Save a clean dataset for analysis
# ------------------------------------------------------------
OUT_DIR = Path("data/cleaned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

panel.to_csv(OUT_DIR / "wdi_gdp_co2_population_panel.csv", index=False)

# ------------------------------------------------------------
# 6) Prints (avoid the “only seeing population” issue)
#    Pandas often truncates columns in terminal output.
# ------------------------------------------------------------
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\nPreview (selected columns):")
cols_show = [
    "country",
    "iso3c",
    "year",
    "gdp_per_capita_const2015_usd",
    "co2_per_capita_tons",
    "population",
]
print(panel[cols_show].head(10))

print("\nColumns:", panel.columns.tolist())
print("\nYears:", panel["year"].min(), "-", panel["year"].max())
print("\nRows:", len(panel))

print("\nNon-missing counts:")
for c in ["gdp_per_capita_const2015_usd", "co2_per_capita_tons", "population"]:
    if c in panel.columns:
        print(f"  {c}: {panel[c].notna().sum()}")
