from __future__ import annotations

from typing import Any, Dict, List

from ingestion.loader.generic_loader import GenericVehicleLoader


class MotorcyclesLoader(GenericVehicleLoader):
    """
    Hero MotoCorp motorcycle loader.

    Product/model names are discovered dynamically.

    No motorcycle model is hard-coded.
    """

    CATEGORY = "motorcycles"

    SOURCES: List[Dict[str, Any]] = [
        {
            "brand": "hero",
            "url": (
                "https://www.heromotocorp.com/"
                "en-in/motorcycles.html"
            ),
            "vehicle_type": "motorcycle",
        }
    ]
