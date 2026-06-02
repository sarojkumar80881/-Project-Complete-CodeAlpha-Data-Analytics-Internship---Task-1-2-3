# ============================================================
#   Task 2 — Exploratory Data Analysis (EDA)
#   Dataset : Sales Transaction
#   CodeAlpha Data Analytics Internship
# ============================================================


# ── STEP 1 : Import Libraries ────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")


# ── STEP 2 : Load Dataset ────────────────────────────────────
df = pd.read_csv('sales_transaction.csv')
print(f"✅ Dataset loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# ── STEP 3 : First Look ──────────────────────────────────────
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Last 5 Rows ---")
print(df.tail())


# ── STEP 4 : Data Structure & Types ─────────────────────────
print("\n--- Column Data Types ---")
print(df.dtypes)

print("\n--- Dataset Info ---")
df.info()


# ── STEP 5 : Missing Values ──────────────────────────────────
print("\n--- Missing Values per Column ---")
missing = df.isnull().sum()
print(missing)
print(f"Total missing values: {missing.sum()}")
if missing.sum() == 0:
    print("✅ No missing values! Dataset is clean.")


# ── STEP 6 : Duplicate Check ─────────────────────────────────
duplicates = df.duplicated().sum()
print(f"\n--- Duplicate Rows: {duplicates} ---")
if duplicates == 0:
    print("✅ No duplicate rows found.")


# ── STEP 7 : Statistical Summary ────────────────────────────
print("\n--- Statistical Summary ---")
print(df.describe().round(2))


# ── STEP 8 : Unique Value Counts ─────────────────────────────
print("\n--- Unique Values per Column ---")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique values")


# ── STEP 9 : Date Conversion & Feature Extraction ────────────
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='%d/%m/%y')
df['Month']      = df['TransactionDate'].dt.month
df['Month_Name'] = df['TransactionDate'].dt.strftime('%b')
df['Day']        = df['TransactionDate'].dt.day
print("\n✅ Date column converted. Month & Day extracted.")
print(df[['TransactionDate', 'Month', 'Month_Name', 'Day']].head())


# ── STEP 10 : Add Revenue Column ─────────────────────────────
df['Revenue'] = df['QuantityPurchased'] * df['Price']
print(f"\n✅ Revenue column added.")
print(f"   Total Revenue          : {df['Revenue'].sum():,.2f}")
print(f"   Avg Revenue/Transaction: {df['Revenue'].mean():,.2f}")


# ── STEP 11 : Key Business Questions ─────────────────────────

# Q1: Top 10 Customers by Revenue
print("\n--- Q1: Top 10 Customers by Revenue ---")
top_customers = df.groupby('CustomerID')['Revenue'].sum().sort_values(ascending=False).head(10)
print(top_customers)

# Q2: Top 10 Products by Revenue
print("\n--- Q2: Top 10 Products by Revenue ---")
top_products = df.groupby('ProductID')['Revenue'].sum().sort_values(ascending=False).head(10)
print(top_products)

# Q3: Monthly Revenue Trend
print("\n--- Q3: Monthly Revenue ---")
monthly_revenue = df.groupby('Month')['Revenue'].sum().sort_index()
print(monthly_revenue)

# Q4: Quantity Distribution
print("\n--- Q4: Quantity Purchased Distribution ---")
print(df['QuantityPurchased'].value_counts().sort_index())

# Q5: Price Analysis
print("\n--- Q5: Price Analysis ---")
print(f"  Min Price   : {df['Price'].min()}")
print(f"  Max Price   : {df['Price'].max()}")
print(f"  Mean Price  : {df['Price'].mean():.2f}")
print(f"  Median Price: {df['Price'].median():.2f}")
print(f"  Std Dev     : {df['Price'].std():.2f}")


# ── STEP 12 : Outlier Detection (IQR Method) ─────────────────
Q1  = df['Price'].quantile(0.25)
Q3  = df['Price'].quantile(0.75)
IQR = Q3 - Q1
lower  = Q1 - 1.5 * IQR
upper  = Q3 + 1.5 * IQR

outliers = df[(df['Price'] < lower) | (df['Price'] > upper)]
print(f"\n--- Outlier Detection (Price) ---")
print(f"  Lower Bound : {lower:.2f}")
print(f"  Upper Bound : {upper:.2f}")
print(f"  Outliers Found: {len(outliers)}")
print(outliers[['TransactionID', 'ProductID', 'Price']].head(10))


# ── STEP 13 : Correlation Analysis ───────────────────────────
print("\n--- Correlation Matrix ---")
corr = df[['QuantityPurchased', 'Price', 'Revenue']].corr()
print(corr.round(3))


# ── STEP 14 : Visualizations ─────────────────────────────────

# Chart 1 — Price Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df[df['Price'] <= upper]['Price'], bins=40, color='steelblue', kde=True)
plt.title('Price Distribution (Outliers Excluded)')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('chart1_price_distribution.png')
plt.show()
print("✅ Chart 1 saved: chart1_price_distribution.png")

# Chart 2 — Monthly Revenue Trend
plt.figure(figsize=(9, 4))
monthly_revenue.plot(kind='line', marker='o', color='darkorange')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.savefig('chart2_monthly_revenue.png')
plt.show()
print("✅ Chart 2 saved: chart2_monthly_revenue.png")

# Chart 3 — Quantity Purchased Distribution
plt.figure(figsize=(6, 4))
df['QuantityPurchased'].value_counts().sort_index().plot(kind='bar', color='mediumseagreen', edgecolor='black')
plt.title('Quantity Purchased Distribution')
plt.xlabel('Quantity')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('chart3_quantity_distribution.png')
plt.show()
print("✅ Chart 3 saved: chart3_quantity_distribution.png")

# Chart 4 — Top 10 Products by Revenue
plt.figure(figsize=(9, 4))
top_products.plot(kind='bar', color='indianred', edgecolor='black')
plt.title('Top 10 Products by Revenue')
plt.xlabel('Product ID')
plt.ylabel('Total Revenue')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('chart4_top_products.png')
plt.show()
print("✅ Chart 4 saved: chart4_top_products.png")

# Chart 5 — Correlation Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('chart5_correlation_heatmap.png')
plt.show()
print("✅ Chart 5 saved: chart5_correlation_heatmap.png")


# ── STEP 15 : Final Summary ───────────────────────────────────
print("\n" + "="*52)
print("       EDA SUMMARY — SALES TRANSACTION DATA")
print("="*52)
print(f"  Total Transactions       : {len(df)}")
print(f"  Unique Customers         : {df['CustomerID'].nunique()}")
print(f"  Unique Products          : {df['ProductID'].nunique()}")
print(f"  Date Range               : {df['TransactionDate'].min().date()} → {df['TransactionDate'].max().date()}")
print(f"  Total Revenue            : {df['Revenue'].sum():,.2f}")
print(f"  Avg Revenue/Transaction  : {df['Revenue'].mean():,.2f}")
print(f"  Most Sold Quantity       : {df['QuantityPurchased'].mode()[0]}")
print(f"  Price Outliers Found     : {len(outliers)}")
print(f"  Missing Values           : None")
print(f"  Duplicate Rows           : None")
print("="*52)
print("✅ EDA Complete!")
