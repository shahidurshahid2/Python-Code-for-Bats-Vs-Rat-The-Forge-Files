# ==============================
# HIT140 - Assessment 3
# Bat vs Rat Data Analysis (Investigation A + B)
# ==============================

# --- Import libraries ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# Seaborn visual style
sns.set(style="whitegrid")

# ------------------------------
# Load the datasets
# ------------------------------
df1 = pd.read_csv("/Users/shahid/Downloads/dataset1.csv")   # Bat landings dataset
df2 = pd.read_csv("/Users/shahid/Downloads/dataset2.csv")   # Rat arrivals dataset

print("Dataset1 shape:", df1.shape)
print("Dataset2 shape:", df2.shape)
print("\nDataset1 preview:")
print(df1.head())
print("\nDataset2 preview:")
print(df2.head())

# ------------------------------
# Add a 'season' column (from month) for both datasets
# ------------------------------
def month_to_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

if "month" in df1.columns:
    df1["season"] = df1["month"].apply(month_to_season)
if "month" in df2.columns:
    df2["season"] = df2["month"].apply(month_to_season)

# ==========================================================
# INVESTIGATION A – Visualisations and Descriptive Statistics
# ==========================================================

# 1. Risk-taking vs Avoidance
plt.figure(figsize=(6,4))
sns.countplot(x="risk", data=df1, palette="Set2", hue=None)
plt.title("Bat Behaviour: Risk-taking vs Avoidance")
plt.xlabel("Risk Behaviour (0 = Avoidance, 1 = Risk-taking)")
plt.ylabel("Count")
plt.show()

# 2. Risk vs Reward
plt.figure(figsize=(6,4))
sns.countplot(x="risk", hue="reward", data=df1, palette="muted")
plt.title("Risk Behaviour and Foraging Reward")
plt.xlabel("Risk Behaviour (0 = Avoidance, 1 = Risk-taking)")
plt.ylabel("Count")
plt.legend(title="Reward (0 = No, 1 = Yes)")
plt.show()

# 3. Seasonal Bat Behaviour
plt.figure(figsize=(6,4))
sns.countplot(x="season", hue="risk", data=df1, palette="Set1")
plt.title("Seasonal Bat Behaviour: Risk vs Avoidance")
plt.xlabel("Season")
plt.ylabel("Number of Landings")
plt.show()

# 4. Rat Arrivals per Season
plt.figure(figsize=(6,4))
sns.barplot(x="season", y="rat_arrival_number", data=df2, estimator=sum, palette="Set3")
plt.title("Total Rat Arrivals per Season")
plt.xlabel("Season")
plt.ylabel("Rat Arrivals")
plt.show()

# 5. Bat Landings vs Rat Arrivals
plt.figure(figsize=(6,4))
sns.scatterplot(x="rat_arrival_number", y="bat_landing_number", data=df2, hue="season", palette="deep")
plt.title("Bat Landings vs Rat Arrivals")
plt.xlabel("Rat Arrivals (per 30 mins)")
plt.ylabel("Bat Landings")
plt.show()

# 6. Distribution of Rat Minutes
plt.figure(figsize=(6,4))
sns.histplot(df2["rat_minutes"], bins=20, kde=True, color="purple")
plt.title("Distribution of Rat Minutes on Platform")
plt.xlabel("Rat Minutes (per 30 mins)")
plt.ylabel("Frequency")
plt.show()

# ==========================================================
# INVESTIGATION B – Inferential & Regression Analyses
# ==========================================================

# --- 7. Seasonal Summary Statistics ---
season_summary = df2.groupby("season")[["rat_arrival_number","bat_landing_number","rat_minutes"]].mean()
print("\nAverage Rat/Bat Activity per Season:\n", season_summary)

# --- 8. Chi-square test (risk vs reward) ---
contingency_table = pd.crosstab(df1["risk"], df1["reward"])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print("\nChi-square test between risk and reward:")
print("Chi2 =", chi2, "p-value =", p)

# --- 9. Simple Linear Regression (bat_landing_number ~ rat_arrival_number) ---
X1 = sm.add_constant(df2["rat_arrival_number"].fillna(0))
y1 = df2["bat_landing_number"].fillna(0)
model1 = sm.OLS(y1, X1).fit()
print("\nSimple Linear Regression Summary:")
print(model1.summary())

# Visualise the regression line
plt.figure(figsize=(6,4))
sns.regplot(x="rat_arrival_number", y="bat_landing_number", data=df2, scatter_kws={"alpha":0.4}, line_kws={"color":"red"})
plt.title("Simple Linear Regression: Bat Landings vs Rat Arrivals")
plt.xlabel("Rat Arrivals (per 30 mins)")
plt.ylabel("Bat Landings")
plt.show()
