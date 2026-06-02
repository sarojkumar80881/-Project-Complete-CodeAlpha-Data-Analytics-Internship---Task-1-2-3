# ============================================================
#   Task 3 — Data Visualization
#   Dataset : Sales Transaction
#   CodeAlpha Data Analytics Internship
# ============================================================


# ── STEP 1 : Import Libraries ────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Global style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11

print("✅ Libraries imported!")


# ── STEP 2 : Load & Prepare Data ─────────────────────────────
df = pd.read_csv('sales_transaction.csv')

df['TransactionDate']  = pd.to_datetime(df['TransactionDate'], format='%d/%m/%y')
df['Month']            = df['TransactionDate'].dt.month
df['Month_Name']       = df['TransactionDate'].dt.strftime('%b')
df['Revenue']          = df['QuantityPurchased'] * df['Price']

print(f"✅ Dataset ready — {len(df)} rows\n")


# ════════════════════════════════════════════════════════════
#  CHART 1 — Monthly Revenue Trend (Line Chart)
# ════════════════════════════════════════════════════════════
monthly = df.groupby(['Month', 'Month_Name'])['Revenue'].sum().reset_index()
monthly = monthly.sort_values('Month')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(monthly['Month_Name'], monthly['Revenue'],
        marker='o', linewidth=2.5, color='#2196F3', markersize=8, label='Monthly Revenue')
ax.fill_between(monthly['Month_Name'], monthly['Revenue'], alpha=0.15, color='#2196F3')

for i, row in monthly.iterrows():
    ax.annotate(f"{row['Revenue']:,.0f}",
                xy=(row['Month_Name'], row['Revenue']),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, color='#333')

ax.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Month')
ax.set_ylabel('Total Revenue')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.legend()
plt.tight_layout()
plt.savefig('chart1_monthly_revenue_trend.png', bbox_inches='tight')
plt.show()
print("✅ Chart 1 saved: chart1_monthly_revenue_trend.png")


# ════════════════════════════════════════════════════════════
#  CHART 2 — Top 10 Products by Revenue (Bar Chart)
# ════════════════════════════════════════════════════════════
top_products = (df.groupby('ProductID')['Revenue']
                  .sum()
                  .sort_values(ascending=False)
                  .head(10)
                  .reset_index())

fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette('Blues_d', len(top_products))
bars = ax.bar(top_products['ProductID'].astype(str),
              top_products['Revenue'], color=colors, edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, top_products['Revenue']):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 500,
            f'{val:,.0f}',
            ha='center', va='bottom', fontsize=8.5)

ax.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Product ID')
ax.set_ylabel('Total Revenue')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.savefig('chart2_top10_products.png', bbox_inches='tight')
plt.show()
print("✅ Chart 2 saved: chart2_top10_products.png")


# ════════════════════════════════════════════════════════════
#  CHART 3 — Price Distribution (Histogram + KDE)
# ════════════════════════════════════════════════════════════

# Remove extreme outliers for better visualization
Q1, Q3 = df['Price'].quantile([0.25, 0.75])
IQR     = Q3 - Q1
price_clean = df[df['Price'] <= Q3 + 1.5 * IQR]['Price']

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(price_clean, bins=40, kde=True, color='#4CAF50',
             edgecolor='white', linewidth=0.5, ax=ax)

ax.axvline(price_clean.mean(),   color='red',    linestyle='--', linewidth=1.5, label=f'Mean: {price_clean.mean():.2f}')
ax.axvline(price_clean.median(), color='orange', linestyle='--', linewidth=1.5, label=f'Median: {price_clean.median():.2f}')

ax.set_title('Price Distribution (Outliers Excluded)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig('chart3_price_distribution.png', bbox_inches='tight')
plt.show()
print("✅ Chart 3 saved: chart3_price_distribution.png")


# ════════════════════════════════════════════════════════════
#  CHART 4 — Quantity Purchased (Pie Chart)
# ════════════════════════════════════════════════════════════
qty_counts = df['QuantityPurchased'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 7))
colors_pie = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350']
wedges, texts, autotexts = ax.pie(
    qty_counts.values,
    labels=[f'Qty {q}' for q in qty_counts.index],
    autopct='%1.1f%%',
    colors=colors_pie,
    startangle=140,
    pctdistance=0.82,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')

ax.set_title('Quantity Purchased Distribution', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('chart4_quantity_pie.png', bbox_inches='tight')
plt.show()
print("✅ Chart 4 saved: chart4_quantity_pie.png")


# ════════════════════════════════════════════════════════════
#  CHART 5 — Top 10 Customers by Revenue (Horizontal Bar)
# ════════════════════════════════════════════════════════════
top_customers = (df.groupby('CustomerID')['Revenue']
                   .sum()
                   .sort_values(ascending=True)
                   .tail(10))

fig, ax = plt.subplots(figsize=(10, 6))
colors_cust = sns.color_palette('Oranges', len(top_customers))
bars = ax.barh(top_customers.index.astype(str), top_customers.values,
               color=colors_cust, edgecolor='white')

for bar, val in zip(bars, top_customers.values):
    ax.text(val + 200, bar.get_y() + bar.get_height() / 2,
            f'{val:,.0f}', va='center', fontsize=9)

ax.set_title('Top 10 Customers by Revenue', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Total Revenue')
ax.set_ylabel('Customer ID')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.savefig('chart5_top10_customers.png', bbox_inches='tight')
plt.show()
print("✅ Chart 5 saved: chart5_top10_customers.png")


# ════════════════════════════════════════════════════════════
#  CHART 6 — Correlation Heatmap
# ════════════════════════════════════════════════════════════
corr = df[['QuantityPurchased', 'Price', 'Revenue']].corr()

fig, ax = plt.subplots(figsize=(7, 5))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=1, linecolor='white',
            annot_kws={'size': 13, 'weight': 'bold'},
            vmin=-1, vmax=1, ax=ax)

ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('chart6_correlation_heatmap.png', bbox_inches='tight')
plt.show()
print("✅ Chart 6 saved: chart6_correlation_heatmap.png")


# ════════════════════════════════════════════════════════════
#  CHART 7 — Revenue by Quantity (Box Plot)
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
revenue_clean = df[df['Revenue'] <= df['Revenue'].quantile(0.95)]
sns.boxplot(data=revenue_clean, x='QuantityPurchased', y='Revenue',
            palette='Set2', width=0.5, ax=ax)

ax.set_title('Revenue Distribution by Quantity Purchased', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Quantity Purchased')
ax.set_ylabel('Revenue')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.savefig('chart7_revenue_by_quantity_boxplot.png', bbox_inches='tight')
plt.show()
print("✅ Chart 7 saved: chart7_revenue_by_quantity_boxplot.png")


# ════════════════════════════════════════════════════════════
#  CHART 8 — Daily Transaction Count (Area Chart)
# ════════════════════════════════════════════════════════════
daily_txn = df.groupby('TransactionDate')['TransactionID'].count().reset_index()
daily_txn.columns = ['Date', 'Transactions']

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(daily_txn['Date'], daily_txn['Transactions'],
                alpha=0.4, color='#9C27B0')
ax.plot(daily_txn['Date'], daily_txn['Transactions'],
        color='#9C27B0', linewidth=1.5)

ax.set_title('Daily Transaction Count Over Time', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Date')
ax.set_ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('chart8_daily_transactions.png', bbox_inches='tight')
plt.show()
print("✅ Chart 8 saved: chart8_daily_transactions.png")


# ════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "="*52)
print("   TASK 3 — VISUALIZATION COMPLETE")
print("="*52)
print("  Chart 1 : Monthly Revenue Trend (Line)")
print("  Chart 2 : Top 10 Products by Revenue (Bar)")
print("  Chart 3 : Price Distribution (Histogram+KDE)")
print("  Chart 4 : Quantity Purchased (Pie)")
print("  Chart 5 : Top 10 Customers by Revenue (H-Bar)")
print("  Chart 6 : Correlation Heatmap")
print("  Chart 7 : Revenue by Quantity (Box Plot)")
print("  Chart 8 : Daily Transaction Count (Area)")
print("="*52)
print("✅ All 8 charts saved as PNG files!")
