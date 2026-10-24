import asyncio
import asyncpg
from config.settings import DB_PARAMS

async def check():
    conn = await asyncpg.connect(**DB_PARAMS)
    try:
        # Count rows per category
        cats = await conn.fetch('SELECT category, COUNT(*) as cnt FROM brochure_chunks GROUP BY category')
        print('=== Row counts by category ===')
        for row in cats:
            print(f"{row['category']}: {row['cnt']} rows")

        # Sample one car
        print('\n=== Sample cars row ===')
        car = await conn.fetchrow("SELECT category, metadata FROM brochure_chunks WHERE category = 'cars' LIMIT 1")
        if car:
            print(f"Category: {car['category']}")
            meta_str = str(car['metadata'])
            print(f"Metadata (first 300 chars): {meta_str[:300]}")
        else:
            print('NO CARS DATA')

        # Sample one motorcycle
        print('\n=== Sample motorcycles row ===')
        moto = await conn.fetchrow("SELECT category, metadata FROM brochure_chunks WHERE category = 'motorcycles' LIMIT 1")
        if moto:
            print(f"Category: {moto['category']}")
            meta_str = str(moto['metadata'])
            print(f"Metadata (first 300 chars): {meta_str[:300]}")
        else:
            print('NO MOTORCYCLES DATA')

        # Check metadata_version
        print('\n=== Latest metadata versions ===')
        versions = await conn.fetch('SELECT DISTINCT metadata_version FROM brochure_chunks ORDER BY metadata_version DESC LIMIT 5')
        for v in versions:
            print(f"  {v['metadata_version']}")
    finally:
        await conn.close()

asyncio.run(check())
