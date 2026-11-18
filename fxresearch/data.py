"""Auditable CSV -> indexed SQLite, with bounded-memory chronological reads.

Timestamps are exact integer nanoseconds. Unknown receive times/sizes/firmness
remain labelled unknown. No missing size is silently replaced by infinity.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import re
import sqlite3


NS = 1_000_000_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes(
 id INTEGER PRIMARY KEY,
 market_ns INTEGER NOT NULL, receive_ns INTEGER NOT NULL,
 venue TEXT NOT NULL, base TEXT NOT NULL, quote TEXT NOT NULL,
 bid REAL NOT NULL, ask REAL NOT NULL,
 bid_size REAL, ask_size REAL, quote_id TEXT NOT NULL,
 receive_assumed INTEGER NOT NULL, firm INTEGER,
 source_id INTEGER NOT NULL, source_row INTEGER NOT NULL,
 UNIQUE(source_id,source_row));
CREATE TABLE IF NOT EXISTS sources(
 id INTEGER PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
 config_json TEXT NOT NULL, rows INTEGER NOT NULL DEFAULT 0);
"""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_ns(value, source_timezone=None, fmt=None, unit=None):
    value = str(value).strip()
    if fmt:
        moment = datetime.strptime(value, fmt)
        remainder = 0
    elif re.fullmatch(r"[+-]?\d+", value):
        multiplier = {"s": NS, "ms": 1_000_000, "us": 1000, "ns": 1}.get(unit)
        if multiplier is None:
            raise ValueError("integer epoch timestamp requires explicit timestamp_unit")
        return int(value) * multiplier
    else:
        match = re.fullmatch(r"(.*T\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)?", value)
        if not match:
            raise ValueError("ISO8601 timestamp or explicit timestamp_format required")
        main, fraction, offset = match.groups()
        moment = datetime.fromisoformat(main + ("+00:00" if offset == "Z" else offset or ""))
        remainder = int((fraction or "").ljust(9, "0"))
    if moment.tzinfo is None:
        if source_timezone is None:
            raise ValueError("naive timestamp requires source_timezone; UTC is not guessed")
        if source_timezone == "EST_FIXED":
            moment = moment.replace(tzinfo=timezone(timedelta(hours=-5)))
        else:
            tz = ZoneInfo(source_timezone)
            first, second = moment.replace(tzinfo=tz, fold=0), moment.replace(tzinfo=tz, fold=1)
            if first.utcoffset() != second.utcoffset():
                raise ValueError("ambiguous/nonexistent DST local time; supply an explicit UTC offset")
            moment = first
    delta = moment.astimezone(timezone.utc) - EPOCH
    return (delta.days * 86400 + delta.seconds) * NS + delta.microseconds * 1000 + remainder


def iso_utc(value):
    seconds, nano = divmod(int(value), NS)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{nano:09d}Z"


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def optional_float(value):
    if value is None or str(value).strip() in ("", "NA", "NaN", "null"):
        return None
    number = float(value)
    if not (number >= 0 and number < float("inf")):
        raise ValueError("sizes must be nonnegative finite amounts or missing")
    return number


def ingest_csv(path, database, config):
    """Idempotent ingestion by hash, atomic transaction, validation is strict.

    One source file has one mapping. Base/quote/venue can be constant metadata
    or columns. Size units are BASE units on both sides, as in a spot book.
    """
    path = Path(path)
    digest = sha256_file(path)
    conn = connect(database)
    existing = conn.execute("SELECT id,rows,config_json FROM sources WHERE sha256=?", (digest,)).fetchone()
    encoded = json.dumps(config, sort_keys=True)
    if existing:
        conn.close()
        if existing[2] != encoded:
            raise ValueError("same source hash with different interpretation; use a separate database")
        return {"path": str(path), "sha256": digest, "rows": existing[1], "reused": True}
    columns = config.get("columns", {})
    def read(row, name, default=None):
        col = columns.get(name)
        return row.get(col, default) if col else config.get(name, default)
    seen = 0
    try:
        with conn:
            source = conn.execute("INSERT INTO sources(path,sha256,config_json) VALUES(?,?,?)",
                                  (str(path), digest, encoded)).lastrowid
            batch = []
            with path.open(newline="", encoding=config.get("encoding", "utf-8-sig")) as stream:
                reader = csv.DictReader(stream, delimiter=config.get("delimiter", ","),
                                        fieldnames=config.get("fieldnames"))
                for seen, row in enumerate(reader, 1):
                    base, quote = str(read(row, "base", "")).upper(), str(read(row, "quote", "")).upper()
                    venue = str(read(row, "venue", "UNKNOWN"))
                    if not base or not quote or base == quote:
                        raise ValueError(f"row {seen}: distinct base and quote required")
                    m = timestamp_ns(read(row, "market_ts"), config.get("source_timezone"), config.get("timestamp_format"), config.get("timestamp_unit"))
                    received = read(row, "receive_ts")
                    assumed = int(received in (None, ""))
                    r = m if assumed else timestamp_ns(received, config.get("receive_timezone", config.get("source_timezone")), config.get("receive_timestamp_format"), config.get("receive_timestamp_unit", config.get("timestamp_unit")))
                    if r < m - int(config.get("clock_skew_tolerance_ns", 0)):
                        raise ValueError(f"row {seen}: receive time precedes market time; clock reconciliation required")
                    bid, ask = float(read(row, "bid")), float(read(row, "ask"))
                    if not (0 < bid <= ask < float("inf")):
                        raise ValueError(f"row {seen}: invalid bid/ask")
                    bid_size, ask_size = optional_float(read(row, "bid_size")), optional_float(read(row, "ask_size"))
                    firm_raw = read(row, "firm")
                    firm_token = str(firm_raw).strip().lower() if firm_raw is not None else ""
                    if firm_token not in ("", "unknown", "1", "true", "yes", "0", "false", "no"):
                        raise ValueError(f"row {seen}: unrecognised firm flag")
                    firm = None if firm_token in ("", "unknown") else int(firm_token in ("1", "true", "yes"))
                    quote_id = str(read(row, "quote_id") or f"{digest[:16]}:{seen}")
                    batch.append((m, r, venue, base, quote, bid, ask, bid_size, ask_size,
                                  quote_id, assumed, firm, source, seen))
                    if len(batch) == 10000:
                        conn.executemany("INSERT INTO quotes(market_ns,receive_ns,venue,base,quote,bid,ask,bid_size,ask_size,quote_id,receive_assumed,firm,source_id,source_row) VALUES(" + ",".join("?" * 14) + ")", batch)
                        batch.clear()
                if batch:
                    conn.executemany("INSERT INTO quotes(market_ns,receive_ns,venue,base,quote,bid,ask,bid_size,ask_size,quote_id,receive_assumed,firm,source_id,source_row) VALUES(" + ",".join("?" * 14) + ")", batch)
            conn.execute("UPDATE sources SET rows=? WHERE id=?", (seen, source))
            conn.execute("CREATE INDEX IF NOT EXISTS quote_receive ON quotes(receive_ns,market_ns,id)")
            conn.execute("CREATE INDEX IF NOT EXISTS quote_market ON quotes(market_ns,id)")
            conn.execute("CREATE INDEX IF NOT EXISTS quote_pair ON quotes(venue,base,quote,market_ns)")
    finally:
        conn.close()
    return {"path": str(path), "sha256": digest, "rows": seen, "reused": False}


@dataclass(frozen=True)
class Quote:
    market_ns: int
    receive_ns: int
    venue: str
    base: str
    quote: str
    bid: float
    ask: float
    bid_size: float | None
    ask_size: float | None
    quote_id: str
    receive_assumed: bool
    firm: bool | None


def iter_quotes(database, start=None, end=None, clock="receive"):
    if clock not in ("receive", "market"):
        raise ValueError("clock must be receive or market")
    conn = sqlite3.connect(database)
    field = clock + "_ns"
    where, args = [], []
    for op, value in ((">=", start), ("<", end)):
        if value is not None:
            where.append(field + op + "?")
            args.append(int(value))
    sql = "SELECT market_ns,receive_ns,venue,base,quote,bid,ask,bid_size,ask_size,quote_id,receive_assumed,firm FROM quotes"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY " + field + ",market_ns,id"
    try:
        for row in conn.execute(sql, args):
            yield Quote(*row)
    finally:
        conn.close()


def dataset_summary(database):
    conn = sqlite3.connect(database)
    total = conn.execute("SELECT COUNT(*),MIN(market_ns),MAX(market_ns),SUM(receive_assumed),SUM(bid_size IS NULL OR ask_size IS NULL),SUM(firm IS NULL),SUM(firm=1) FROM quotes").fetchone()
    instruments = conn.execute("SELECT venue,base,quote,COUNT(*) FROM quotes GROUP BY venue,base,quote").fetchall()
    sources = conn.execute("SELECT path,sha256,rows FROM sources ORDER BY id").fetchall()
    conn.close()
    return {"rows": total[0], "start_market_utc": iso_utc(total[1]) if total[1] is not None else None,
            "end_market_utc": iso_utc(total[2]) if total[2] is not None else None,
            "receive_times_assumed": total[3] or 0, "rows_without_sizes": total[4] or 0,
            "rows_firm_unknown": total[5] or 0, "rows_firm_true": total[6] or 0,
            "instruments": instruments, "sources": sources}
