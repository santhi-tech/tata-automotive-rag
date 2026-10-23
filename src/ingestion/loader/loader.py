from __future__ import annotations

from ingestion.loader.cars_loader import CarsLoader
from ingestion.loader.motorcycles_loader import MotorcyclesLoader
from ingestion.loader.scooters_loader import ScootersLoader
from ingestion.loader.spare_parts_loader import SparePartsLoader

LOADERS = {"cars": CarsLoader, "motorcycles": MotorcyclesLoader, "scooters": ScootersLoader, "spare_parts": SparePartsLoader}
