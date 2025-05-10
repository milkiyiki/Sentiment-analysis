from google_play_scraper import Sort, reviews
import pandas as pd
import time

# Konfigurasi scraping
app_package = 'com.ubercab'  # Package name Uber
lang = 'en'  # Bahasa Inggris
country = 'us'  # Server US
count_total = 10000  # Target jumlah data
batch_size = 100  # Ambil 100 per request

results = []
next_token = None

print("Starting scraping Uber reviews...")

while len(results) < count_total:
    count_to_fetch = min(batch_size, count_total - len(results))
    
    review_batch, next_token = reviews(
        app_package,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        count=count_to_fetch,
        continuation_token=next_token
    )
    
    results.extend(review_batch)
    
    print(f"Scraped {len(results)} reviews...")

    if not next_token:
        break  # Tidak ada lagi review
    
    time.sleep(1)  # Hindari terlalu cepat request

# Simpan ke CSV
df = pd.DataFrame(results)
df.to_csv('uber_reviews_en.csv', index=False)
print("Scraping selesai. Data disimpan ke 'uber_reviews_en.csv'")
