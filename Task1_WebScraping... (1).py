# ============================================================
#   Task 1 — Web Scraping
#   Source  : Books to Scrape (books.toscrape.com)
#             A free, legal website made for scraping practice
#   CodeAlpha Data Analytics Internship
# ============================================================


# ── STEP 1 : Install & Import Libraries ──────────────────────
# Run this line first in Colab if needed:
# !pip install requests beautifulsoup4

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully!")


# ── STEP 2 : Define Target Website ───────────────────────────
BASE_URL = "https://books.toscrape.com/catalogue/"
START_URL = "https://books.toscrape.com/catalogue/page-1.html"

print(f"🌐 Target Website : books.toscrape.com")
print(f"   Purpose        : Scrape book titles, prices, ratings & availability")


# ── STEP 3 : Helper — Convert Rating Word to Number ──────────
def rating_to_number(word):
    mapping = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    return mapping.get(word, 0)


# ── STEP 4 : Scrape Multiple Pages ───────────────────────────
all_books = []
current_url = START_URL
page_num = 1
MAX_PAGES = 10          # scraping 10 pages = ~200 books

print(f"\n🔄 Starting scrape — collecting up to {MAX_PAGES} pages...\n")

while current_url and page_num <= MAX_PAGES:
    try:
        response = requests.get(current_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        books = soup.find_all('article', class_='product_pod')

        for book in books:
            # Title
            title = book.h3.a['title']

            # Price — remove £ symbol
            price_text = book.find('p', class_='price_color').text.strip()
            price = float(price_text.replace('£', '').replace('Â', '').strip())

            # Rating
            rating_word = book.find('p', class_='star-rating')['class'][1]
            rating = rating_to_number(rating_word)

            # Availability
            availability = book.find('p', class_='instock availability').text.strip()

            all_books.append({
                'Title'       : title,
                'Price (£)'   : price,
                'Rating'      : rating,
                'Availability': availability
            })

        print(f"  ✅ Page {page_num} scraped — {len(books)} books collected")

        # Find next page
        next_btn = soup.find('li', class_='next')
        if next_btn:
            next_page = next_btn.a['href']
            current_url = BASE_URL + next_page
            page_num += 1
            time.sleep(0.5)   # polite delay — don't hammer the server
        else:
            current_url = None

    except Exception as e:
        print(f"  ❌ Error on page {page_num}: {e}")
        break

print(f"\n✅ Scraping complete! Total books collected: {len(all_books)}")


# ── STEP 5 : Create DataFrame ────────────────────────────────
df = pd.DataFrame(all_books)
print("\n--- First 5 rows ---")
print(df.head())


# ── STEP 6 : Basic Data Info ─────────────────────────────────
print("\n--- Dataset Shape ---")
print(f"  Rows    : {df.shape[0]}")
print(f"  Columns : {df.shape[1]}")

print("\n--- Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Statistical Summary ---")
print(df.describe().round(2))


# ── STEP 7 : Data Insights ───────────────────────────────────
print("\n--- Key Insights ---")
print(f"  Most Expensive Book : {df.loc[df['Price (£)'].idxmax(), 'Title']} — £{df['Price (£)'].max()}")
print(f"  Cheapest Book       : {df.loc[df['Price (£)'].idxmin(), 'Title']} — £{df['Price (£)'].min()}")
print(f"  Average Price       : £{df['Price (£)'].mean():.2f}")
print(f"  5-Star Books        : {len(df[df['Rating'] == 5])}")
print(f"  In Stock Books      : {len(df[df['Availability'] == 'In stock'])}")

print("\n--- Books per Rating ---")
print(df['Rating'].value_counts().sort_index())


# ── STEP 8 : Save to CSV ─────────────────────────────────────
df.to_csv('scraped_books.csv', index=False)
print("\n✅ Data saved to: scraped_books.csv")


# ── STEP 9 : Visualizations ──────────────────────────────────
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120

# Chart 1 — Price Distribution
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(df['Price (£)'], bins=30, color='steelblue', kde=True, ax=ax)
ax.axvline(df['Price (£)'].mean(), color='red', linestyle='--',
           linewidth=1.5, label=f"Mean: £{df['Price (£)'].mean():.2f}")
ax.set_title('Price Distribution of Scraped Books', fontsize=13, fontweight='bold')
ax.set_xlabel('Price (£)')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig('ws_chart1_price_distribution.png', bbox_inches='tight')
plt.show()
print("✅ Chart 1 saved: ws_chart1_price_distribution.png")

# Chart 2 — Rating Distribution
fig, ax = plt.subplots(figsize=(8, 4))
rating_counts = df['Rating'].value_counts().sort_index()
colors = sns.color_palette('YlOrRd', len(rating_counts))
ax.bar([f'{r} ⭐' for r in rating_counts.index], rating_counts.values,
       color=colors, edgecolor='white')
for i, v in enumerate(rating_counts.values):
    ax.text(i, v + 0.5, str(v), ha='center', fontsize=10, fontweight='bold')
ax.set_title('Number of Books per Rating', fontsize=13, fontweight='bold')
ax.set_xlabel('Rating')
ax.set_ylabel('Number of Books')
plt.tight_layout()
plt.savefig('ws_chart2_rating_distribution.png', bbox_inches='tight')
plt.show()
print("✅ Chart 2 saved: ws_chart2_rating_distribution.png")

# Chart 3 — Average Price per Rating
fig, ax = plt.subplots(figsize=(8, 4))
avg_price = df.groupby('Rating')['Price (£)'].mean()
ax.plot(avg_price.index, avg_price.values, marker='o', color='darkorange',
        linewidth=2.5, markersize=9)
for x, y in zip(avg_price.index, avg_price.values):
    ax.annotate(f'£{y:.2f}', xy=(x, y), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=9)
ax.set_title('Average Price by Rating', fontsize=13, fontweight='bold')
ax.set_xlabel('Star Rating')
ax.set_ylabel('Average Price (£)')
ax.set_xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig('ws_chart3_avg_price_by_rating.png', bbox_inches='tight')
plt.show()
print("✅ Chart 3 saved: ws_chart3_avg_price_by_rating.png")


# ── STEP 10 : Final Summary ───────────────────────────────────
print("\n" + "="*52)
print("   TASK 1 — WEB SCRAPING COMPLETE")
print("="*52)
print(f"  Website Scraped : books.toscrape.com")
print(f"  Pages Scraped   : {page_num - 1}")
print(f"  Total Books     : {len(df)}")
print(f"  Columns Created : {', '.join(df.columns.tolist())}")
print(f"  CSV Saved       : scraped_books.csv")
print(f"  Charts Saved    : 3 PNG files")
print("="*52)
print("✅ Task 1 Complete!")
