"""Lightweight utilities for fetching and loading NYC MTA bus datasets."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import geopandas as gpd
import pandas as pd
from sodapy import Socrata

from src.config import SOCRATA_DOMAIN, SOCRATA_TIMEOUT


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@dataclass(frozen=True)
class DatasetConfig:
    """Structure describing how to fetch a Socrata dataset."""

    dataset_id: str
    base_name: str
    query: str
    limit: int = 50_000


API_DATASETS: Dict[str, DatasetConfig] = {
    "bus_segment_speed_2025": DatasetConfig(
        dataset_id="kufs-yh3x",
        base_name="MTA_Bus_Route_Segment_Speeds_Beginning_2025",
        query="SELECT * WHERE Borough = 'Manhattan' LIMIT {limit} OFFSET {offset}",
        limit=100_000,
    ),
    "bus_segment_speed_2023_2024": DatasetConfig(
        dataset_id="58t6-89vi",
        base_name="MTA_Bus_Route_Segment_Speeds_2023_2024",
        query="SELECT * WHERE Borough = 'Manhattan' LIMIT {limit} OFFSET {offset}",
        limit=100_000,
    ),
    "bus_speed_2025": DatasetConfig(
        dataset_id="4u4b-jge6",
        base_name="MTA_Bus_Speeds_Beginning_2025",
        query="SELECT * WHERE Borough = 'Manhattan' LIMIT {limit} OFFSET {offset}",
        limit=1_000,
    ),
    "bus_speed_2023_2024": DatasetConfig(
        dataset_id="6ksi-7cxr",
        base_name="MTA_Bus_Speeds_2023_2024",
        query=(
            "SELECT * WHERE Borough = 'Manhattan' "
            "AND month BETWEEN '2023-01-01T00:00:00' AND '2024-12-31T23:59:59' "
            "LIMIT {limit} OFFSET {offset}"
        ),
        limit=1_000,
    ),
    "hourly_ridership_2025": DatasetConfig(
        dataset_id="gxb3-akrn",
        base_name="MTA_Bus_Hourly_Ridership_Beginning_2025",
        query=(
            "SELECT transit_timestamp, bus_route, sum(ridership) as total_ridership "
            "WHERE caseless_starts_with(bus_route, 'M') "
            "GROUP BY transit_timestamp, bus_route "
            "LIMIT {limit} OFFSET {offset}"
        ),
        limit=100_000,
    ),
    "hourly_ridership_2023_2024": DatasetConfig(
        dataset_id="kv7t-n8in",
        base_name="MTA_Bus_Hourly_Ridership_2023_2024",
        query=(
            "SELECT transit_timestamp, bus_route, sum(ridership) as total_ridership "
            "WHERE caseless_starts_with(bus_route, 'M') "
            "AND transit_timestamp BETWEEN '2023-01-01T00:00:00' AND '2024-12-31T23:59:59' "
            "GROUP BY transit_timestamp, bus_route "
            "LIMIT {limit} OFFSET {offset}"
        ),
        limit=100_000,
    ),
    "hourly_crossings_2023_2025": DatasetConfig(
        dataset_id="ebfx-2m7v",
        base_name="MTA_Bus_Hourly_Crossings_2023_2025",
        query=(
            "SELECT date, hour, facility, direction, sum(traffic_count) as total_count "
            "WHERE date BETWEEN '2023-01-01T00:00:00' AND '2025-12-31T23:59:59' "
            "GROUP BY date, hour, facility, direction "
            "LIMIT {limit} OFFSET {offset}"
        ),
        limit=100_000,
    ),
    "crz_entries_2023_2025": DatasetConfig(
        dataset_id="t6yz-b64h",
        base_name="MTA_CRZ_Hourly_Entries_2023_2025",
        query="SELECT * LIMIT {limit} OFFSET {offset}",
        limit=100_000,
    ),
    "cbd_vehicle_speeds_2023_2025": DatasetConfig(
        dataset_id="6p29-6xqn",
        base_name="MTA_CBD_Vehicle_Speeds_2023_2025",
        query=(
            "SELECT * WHERE Month BETWEEN '2023-01-01T00:00:00' AND '2025-12-31T23:59:59' "
            "LIMIT {limit} OFFSET {offset}"
        ),
        limit=100,
    ),
    "cbd_bus_routes_2025": DatasetConfig(
        dataset_id="cgzt-smqf",
        base_name="MTA_CBD_Bus_Routes_2025",
        query="SELECT * LIMIT {limit} OFFSET {offset}",
        limit=1_000,
    ),
}


LOCAL_FILES: Dict[str, str] = {
    "stop_data": "manhattan_stops_flat.csv",
    "cbd_geojson_area_2024": "MTA_Central_Business_District_Geofence__Beginning_June_2024_20251105.geojson",
}


DATA_KEY_MAPPING: Dict[str, str] = {
    "bus_segment_speed_2025": "bus_speed_seg_2025",
    "bus_segment_speed_2023_2024": "bus_speed_seg_2023_2024",
    "bus_speed_2025": "bus_speed_2025",
    "bus_speed_2023_2024": "bus_speed_2023_2024",
    "hourly_ridership_2025": "hourly_ridership_2025",
    "hourly_ridership_2023_2024": "hourly_ridership_2023_2024",
    "hourly_crossings_2023_2025": "hourly_crossings_2023_2025",
    "crz_entries_2023_2025": "crz_entries_2023_2025",
    "cbd_vehicle_speeds_2023_2025": "cbd_vehicle_speeds_2023_2025",
    "cbd_bus_routes_2025": "cbd_bus_routes_2025",
}


def _ensure_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_cached_file(base_name: str, data_dir: Path) -> Optional[Path]:
    matches = sorted(data_dir.glob(f"{base_name}_*.csv"), reverse=True)
    return matches[0] if matches else None


def _download_dataset(client: Socrata, config: DatasetConfig, destination: Path) -> pd.DataFrame:
    logger.info("Downloading %s", config.dataset_id)
    chunks = []
    offset = 0

    while True:
        query = config.query.format(limit=config.limit, offset=offset)
        records = client.get(config.dataset_id, query=query)
        chunk = pd.DataFrame.from_records(records)
        if chunk.empty:
            break

        chunks.append(chunk)
        offset += config.limit
        logger.debug("Fetched %s rows (offset=%s)", len(chunk), offset)

        if config.limit >= 100_000:
            time.sleep(1)

    if not chunks:
        logger.warning("No data returned for %s", config.dataset_id)
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(destination, index=False)
    return df


def get_all_mta_data(
    client: Socrata,
    dataset_id: str,
    query_template: str,
    limit: int,
    filename: Optional[str] = None,
) -> pd.DataFrame:
    
    """Backward compatible helper retained for older notebooks/scripts."""

    config = DatasetConfig(dataset_id=dataset_id, base_name="", query=query_template, limit=limit)
    df = _download_dataset(client, config, Path(filename) if filename else Path("/tmp/unused.csv"))
    if filename is None:
        return df
    return pd.read_csv(filename)


def fetch_or_load_data(
    data_dir: str | Path,
    config: DatasetConfig,
    client: Optional[Socrata] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    
    data_path = _ensure_path(data_dir)
    cached = _find_cached_file(config.base_name, data_path)

    if cached and not force_refresh:
        logger.info("Loading cached %s from %s", config.base_name, cached.name)
        return pd.read_csv(cached)

    if client is None:
        raise ValueError(
            "No cached file for %s and no Socrata client available. "
            "Pass an app token or download the CSV manually." % config.base_name
        )

    timestamp = datetime.now().strftime("%Y%m%d")
    destination = data_path / f"{config.base_name}_{timestamp}.csv"
    return _download_dataset(client, config, destination)


def fetch_api_datasets(
    client: Optional[Socrata],
    data_dir: str | Path = "data/raw",
    datasets: Optional[Iterable[str]] = None,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    if datasets is None:
        datasets = API_DATASETS.keys()

    results: Dict[str, pd.DataFrame] = {}
    for dataset_key in datasets:
        config = API_DATASETS.get(dataset_key)
        if config is None:
            logger.warning("Unknown dataset '%s'", dataset_key)
            continue

        df = fetch_or_load_data(data_dir, config, client=client, force_refresh=force_refresh)
        output_key = DATA_KEY_MAPPING.get(dataset_key, dataset_key)
        results[output_key] = df

    return results


def load_local_files(data_dir: str | Path = "data/raw") -> Dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    data_path = _ensure_path(data_dir)
    results: Dict[str, pd.DataFrame | gpd.GeoDataFrame] = {}

    for key, filename in LOCAL_FILES.items():
        file_path = data_path / filename
        if not file_path.exists():
            logger.warning("Local file %s missing at %s", key, file_path)
            results[key] = gpd.GeoDataFrame() if filename.endswith(".geojson") else pd.DataFrame()
            continue

        logger.info("Loading %s", file_path.name)
        if filename.endswith(".geojson"):
            results[key] = gpd.read_file(file_path)
        else:
            results[key] = pd.read_csv(file_path)

    return results


def fetch_all_data(
    app_token: Optional[str] = None,
    data_dir: str | Path = "data/raw",
    datasets: Optional[Iterable[str]] = None,
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    client: Optional[Socrata] = None
    try:
        if app_token:
            client = Socrata(SOCRATA_DOMAIN, app_token=app_token, timeout=SOCRATA_TIMEOUT)
    except Exception as exc:
        logger.error("Could not create Socrata client: %s", exc)
        client = None

    try:
        api_data = fetch_api_datasets(client, data_dir, datasets=datasets, force_refresh=force_refresh)
        local_data = load_local_files(data_dir)
        return {**api_data, **local_data}
    finally:
        if client is not None:
            client.close()


def load_cached_dataset(base_name: str, data_dir: str | Path = "data/raw") -> Optional[pd.DataFrame]:
    data_path = _ensure_path(data_dir)
    cached = _find_cached_file(base_name, data_path)
    if not cached:
        return None
    logger.info("Loading cached dataset %s", cached.name)
    return pd.read_csv(cached)
