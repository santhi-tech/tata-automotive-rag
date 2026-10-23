from __future__ import annotations

from typing import Any, Dict, List

from ingestion.loader.generic_loader import GenericVehicleLoader


class ScootersLoader(GenericVehicleLoader):
    """Hero MotoCorp scooter loader. Scooter model names are discovered dynamically."""

    CATEGORY = "scooters"

    SOURCES: List[Dict[str, Any]] = [
        {
            "brand": "hero",
            "url": "https://www.heromotocorp.com/en-in/scooters.html",
            "vehicle_type": "scooter",
        }
    ]
