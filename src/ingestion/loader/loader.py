from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from ingestion.loader.cars_loader import CarsLoader
from ingestion.loader.motorcycles_loader import MotorcyclesLoader
from ingestion.loader.scooters_loader import ScootersLoader
from ingestion.loader.spare_parts_loader import SparePartsLoader

LOADERS = {"cars": CarsLoader, "motorcycles": MotorcyclesLoader, "scooters": ScootersLoader, "spare_parts": SparePartsLoader}


async def run_loader(name: str, loader) -> list[dict]:
    print(f"\n=== {name} ===")
    results = await loader.download_and_extract()
    print(f"{name}: extracted {len(results)} records")
    return results


async def main() -> None:
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    all_results.extend(
        await run_loader("cars", CarsLoader())
    )

    output_path = output_dir / "vehicle_page_ingestion.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            all_results,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved {len(all_results)} records -> {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    asyncio.run(main())
