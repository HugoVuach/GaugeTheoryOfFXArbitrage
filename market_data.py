"""Acquisition HistData et échantillonnage causal de cotations bid/ask."""
import argparse
import csv
import hashlib
import http.cookiejar
from http.client import HTTPException
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

import config as cfg

HOST = "https://www.histdata.com"
EST_FIXED = timezone(timedelta(hours=-5))
USER_AGENT = "Mozilla/5.0 (compatible; AcademicFXReplay/1.0)"


def timestamp_ms(day, hms):
    return int(datetime.fromisoformat(f"{day}T{hms}+00:00").timestamp() * 1000)


def archive_specs():
    months = sorted({d.strftime("%Y%m") for d in cfg.CALIBRATION + cfg.EVALUATION})
    return [(month, "".join(pair)) for month in months for pair in cfg.PAIRS]


def archive_url(month, symbol):
    return (f"{HOST}/download-free-forex-data/?/ascii/tick-data-quotes/"
            f"{symbol}/{month[:4]}/{int(month[4:])}")


def csv_name(month, symbol):
    return f"DAT_ASCII_{symbol}_T_{month}.csv"


def archive_path(root, month, symbol):
    return root / f"{month}_{symbol}.zip"


def download(root, month, symbol):
    dest = archive_path(root, month, symbol)
    if dest.is_file():
        with ZipFile(dest) as archive:
            archive.getinfo(csv_name(month, symbol))
        return dest
    page = archive_url(month, symbol)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    last_error = None
    for pause in (0, 1, 3):
        time.sleep(pause)
        try:
            opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
            with opener.open(Request(page, headers={"User-Agent": USER_AGENT}), timeout=40) as response:
                html = response.read().decode("utf-8")
            form = html.split('<form id="file_down"', 1)[1].split("</form>", 1)[0]
            fields = dict(re.findall(r'name="(\w+)"[^>]*?value="([^"]+)"', form))
            if set(fields) != {"tk", "date", "datemonth", "platform", "timeframe", "fxpair"}:
                raise ValueError("Formulaire HistData inattendu")
            if (fields["datemonth"], fields["fxpair"]) != (month, symbol):
                raise ValueError("Mois ou instrument inattendu dans le formulaire HistData")
            request = Request(
                f"{HOST}/get.php", data=urlencode(fields).encode(),
                headers={
                    "User-Agent": USER_AGENT,
                    "Referer": page,
                    "Origin": HOST,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            with opener.open(request, timeout=90) as response, part.open("wb") as output:
                shutil.copyfileobj(response, output)
            with ZipFile(part) as archive:
                archive.getinfo(csv_name(month, symbol))
            part.replace(dest)
            print(f"HistData : {dest.name} ({dest.stat().st_size:,} octets)", flush=True)
            return dest
        except (OSError, ValueError, IndexError, KeyError, HTTPException, BadZipFile) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
    raise RuntimeError(f"Échec du téléchargement {page}. Téléchargement manuel vers {dest}.") from last_error


def fetch_all(root):
    with ThreadPoolExecutor(max_workers=3) as executor:
        return list(executor.map(lambda spec: download(root, *spec), archive_specs()))


def manifest(root, output):
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("archive", "sha256", "octets", "page_source"))
        for month, symbol in archive_specs():
            path = archive_path(root, month, symbol)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            writer.writerow((path.name, digest, path.stat().st_size, archive_url(month, symbol)))


def _histdata_stamp(ms):
    instant = datetime.fromtimestamp(ms / 1000, EST_FIXED)
    return instant.strftime("%Y%m%d %H%M%S%f")[:18].encode("ascii")


def extract_archives(root):
    """Conserve les ticks dans les fenêtres d'étude, avec marges avant/après."""
    dates = cfg.CALIBRATION + cfg.EVALUATION
    buckets = {day: {} for day in dates}
    audit = []
    pad = max(cfg.DELAYS_MS) * len(cfg.CURRENCIES)
    for month, symbol in archive_specs():
        days = [day for day in dates if day.strftime("%Y%m") == month]
        windows = {}
        for day in days:
            start = timestamp_ms(day, cfg.START_UTC) - cfg.WARMUP_MS
            end = timestamp_ms(day, cfg.END_UTC) + pad
            lower, upper = _histdata_stamp(start), _histdata_stamp(end)
            windows[lower[:8]] = (day, lower, upper)
        rows = {day: [] for day in days}
        with ZipFile(archive_path(root, month, symbol)) as archive:
            with archive.open(csv_name(month, symbol)) as stream:
                for line in stream:
                    window = windows.get(line[:8])
                    if window is not None and window[1] <= line[:18] <= window[2]:
                        stamp, bid, ask, volume = line.split(b",")
                        rows[window[0]].append((stamp.decode("ascii"), float(bid), float(ask), float(volume)))
        for day, records in rows.items():
            if not records:
                raise ValueError(f"{day} {symbol}: aucun tick dans la fenêtre")
            frame = pd.DataFrame.from_records(records, columns=["stamp", "bid", "ask", "volume"])
            timestamps = pd.to_datetime(frame["stamp"], format="%Y%m%d %H%M%S%f")
            frame["utc_ms"] = (timestamps.to_numpy(dtype="datetime64[ms]").astype(np.int64)
                               + 5 * 3_600_000)
            b = frame["bid"].to_numpy()
            a = frame["ask"].to_numpy()
            invalid = ~np.isfinite(b) | ~np.isfinite(a) | (b <= 0) | (a < b)
            frame.loc[invalid, ["bid", "ask"]] = np.nan
            frame = frame.sort_values("utc_ms", kind="stable").drop_duplicates("utc_ms", keep="last")
            buckets[day][symbol] = (
                frame["utc_ms"].to_numpy(), frame["bid"].to_numpy(), frame["ask"].to_numpy()
            )
            audit.append({
                "date": str(day), "paire": symbol, "lignes": len(records), "ticks_uniques": len(frame),
                "prix_invalides": int(invalid.sum()),
                "volume_non_nul": sum(record[3] != 0 for record in records),
            })
    return buckets, pd.DataFrame(audit)


def decision_grid(day):
    return np.arange(
        timestamp_ms(day, cfg.START_UTC), timestamp_ms(day, cfg.END_UTC),
        cfg.STEP_MS, dtype=np.int64,
    )


def asof(tick_times, bid, ask, query):
    """Dernière mise à jour à ou avant query, avec son âge en ms."""
    index = np.searchsorted(tick_times, query, side="right") - 1
    safe = np.maximum(index, 0)
    available = index >= 0
    age = np.where(available, query - tick_times[safe], np.inf)
    return np.where(available, bid[safe], np.nan), np.where(available, ask[safe], np.nan), age


def quotes_at(data, grid, extra_lag=None):
    """Deux matrices bid/ask et une matrice d'âges, une colonne par paire."""
    lags = {} if extra_lag is None else extra_lag
    columns = []
    for pair in cfg.PAIRS:
        symbol = "".join(pair)
        lag = lags.get(symbol, 0)
        bid, ask, age = asof(*data[symbol], grid - lag)
        columns.append((bid, ask, age + lag))
    return tuple(np.column_stack(column) for column in zip(*columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("raw"))
    args = parser.parse_args()
    fetch_all(args.raw)
