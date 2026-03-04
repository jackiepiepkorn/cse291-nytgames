"""
use this to scrape both boards + words from strandsgame.me
==========================================
POST /ajax-request/get-strand-game-data  {type, date, is_encrypt: true}
XOR-decrypt response fields with key "25" to extract board + metadata.

Field map (obfuscated key -> meaning):
  SBZ5703V  -> board rows (8 entries, each 6 chars)
  WOX734AM  -> word list array
    WO311FDE  -> word text (encrypted)
    IS8BAH13  -> "true" if spangram, "false" if theme word
    CO98FW03  -> word path as JSON [[row,col], ...]
  DASU5E70  -> date string (YYYY-MM-DD)
  CL81443J  -> clue / theme hint
  RC81443J  -> secondary description
  SNB1F072  -> puzzle number

Usage:
    pip install requests
    python strands_datascraper.py                            # full archive
    python strands_datascraper.py --start 2024-03-04 --end 2024-06-30
    python strands_datascraper.py --date 2026-03-02          # single date
    python strands_datascraper.py --validate                 # check existing CSV

Output CSV:
    puzzle_id, date, clue, spanagram, theme_words,
    board_row_0 .. board_row_7,   <- 6 uppercase letters each
    board_flat                    <- all 48 letters

Access board in Python:
    board = get_board_2d(row)   # board[row][col], uppercase
    board[0][0]  # top-left
    board[7][5]  # bottom-right
"""

import argparse
import csv
import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests

POST_URL   = "https://strandsgame.me/ajax-request/get-strand-game-data"
START_DATE = date(2024, 3, 4)
OUTPUT_CSV = "nytgames/data/strands.csv"
XOR_KEY    = "25"

CSV_COLUMNS = [
    "puzzle_id", "date", "clue", "spanagram", "theme_words", "num_theme_words",
    "board_row_0", "board_row_1", "board_row_2", "board_row_3",
    "board_row_4", "board_row_5", "board_row_6", "board_row_7",
    "board_flat",
]

HEADERS = {
    "User-Agent":   "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
    "Referer":      "https://strandsgame.me/nyt-strands-archive",
    "Origin":       "https://strandsgame.me",
    "Accept":       "application/json, */*",
}


# ── Decryption ────────────────────────────────────────────────────────────────

def xor_decrypt(s: str, key: str = XOR_KEY) -> str:
    """XOR each character with cycling key characters."""
    return "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(s))


# ── Parse API response ────────────────────────────────────────────────────────

def parse_response(raw: dict) -> dict | None:
    """
    decrypts and parses a raw API response into our standard row dict
    returns None if board data is no good 
    """
    # ── Board rows (SBZ5703V) ─────────────────────────────────────────────────
    board_enc = raw.get("SBZ5703V", [])
    if not board_enc or len(board_enc) != 8:
        return None

    board = []
    for enc_row in board_enc:
        dec_row = xor_decrypt(enc_row)
        if len(dec_row) != 6:
            return None   # malformed row — shouldn't happen in real responses
        board.append(dec_row.upper())

    # ── Words (WOX734AM) ──────────────────────────────────────────────────────
    word_entries = raw.get("WOX734AM", [])
    spanagram   = ""
    theme_words = []

    for entry in word_entries:
        word      = xor_decrypt(entry.get("WO311FDE", ""))
        is_span   = xor_decrypt(entry.get("IS8BAH13", ""))  # "true" or "false"
        if is_span.lower() == "true":
            spanagram = word.upper()
        else:
            theme_words.append(word.upper())

    # ── Clue (CL81443J) ───────────────────────────────────────────────────────
    clue = xor_decrypt(raw.get("CL81443J", ""))

    # ── Puzzle number (SNB1F072) ──────────────────────────────────────────────
    puzzle_id = xor_decrypt(raw.get("SNB1F072", ""))

    # ── Date (DASU5E70) ───────────────────────────────────────────────────────
    date_str = xor_decrypt(raw.get("DASU5E70", ""))

    return {
        "puzzle_id":   puzzle_id,
        "date":        date_str,
        "clue":        clue,
        "spanagram":   spanagram,
        "theme_words": "|".join(theme_words),
        "num_theme_words": len(theme_words),
        "board_row_0": board[0],
        "board_row_1": board[1],
        "board_row_2": board[2],
        "board_row_3": board[3],
        "board_row_4": board[4],
        "board_row_5": board[5],
        "board_row_6": board[6],
        "board_row_7": board[7],
        "board_flat":  "".join(board),
    }


def get_board_2d(row: dict) -> list[list[str]]:
    """
    convert a CSV row into board[row][col] — all uppercase
    since that is what we use in our strands config 

    Example:
        import csv
        with open("strands_boards.csv") as f:
            rows = list(csv.DictReader(f))
        board = get_board_2d(rows[0])
        print(board[0])     # ['C', 'O', 'W', 'E', 'S', 'S']
        print(board[0][0])  # 'C'  (top-left)
        print(board[7][5])  # bottom-right letter
    """
    return [list(row[f"board_row_{i}"]) for i in range(8)]


# for fetching a single date 
def fetch_date(d: date, session: requests.Session) -> dict | None:
    payload = {
        "type":       "date",
        "date":       d.isoformat(),
        "is_encrypt": True,
    }
    try:
        resp = session.post(POST_URL, json=payload, headers=HEADERS, timeout=12)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        raw = resp.json()
        return parse_response(raw)
    except Exception as e:
        raise RuntimeError(str(e))


# for bulk scraping a bunch/all dates 
def bulk_scrape(start: date, end: date, output_path: Path, delay: float):
    existing  = _load_existing(output_path)
    new_total = (end - start).days + 1
    print(f"Range    : {start} → {end}  ({new_total} dates)")
    print(f"Saved    : {len(existing)} already in {output_path}")
    print(f"Output   : {output_path}\n")

    ok = fail = skip = 0
    first_write = not output_path.exists() or output_path.stat().st_size == 0

    with requests.Session() as session, \
         open(output_path, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if first_write:
            writer.writeheader()

        current = start
        while current <= end:
            ds = current.isoformat()

            if ds in existing:
                skip += 1
                current += timedelta(days=1)
                continue

            print(f"  {ds}  ", end="", flush=True)
            try:
                row = fetch_date(current, session)
            except RuntimeError as e:
                print(f"ERROR: {e}")
                fail += 1
                current += timedelta(days=1)
                time.sleep(delay)
                continue

            if row:
                writer.writerow(row)
                f.flush()
                ok += 1
                flat    = row["board_flat"]
                # Pretty print board inline
                rows_str = " | ".join(flat[i*6:(i+1)*6] for i in range(8))
                print(f"✓  #{row['puzzle_id']:>3}  {rows_str}  span={row['spanagram']}")
            else:
                print("404 / no puzzle")
                skip += 1

            time.sleep(delay)
            current += timedelta(days=1)

    print(f"\n{'='*60}")
    print(f"OK: {ok}  |  Failed: {fail}  |  Skipped: {skip}")
    validate(output_path)


# for returning all the data for that single date 
def fetch_single(d: date, output_path: Path):
    print(f"Fetching {d.isoformat()} ...")
    with requests.Session() as s:
        row = fetch_date(d, s)
    if not row:
        print("No data returned (404 or parse failure).")
        return

    print(f"\nPuzzle #{row['puzzle_id']}  —  {row['date']}")
    print(f"Clue      : {row['clue']}")
    print(f"Spanagram : {row['spanagram']}")
    print(f"Theme words: {row['theme_words']}")
    print(f"Number of Theme words: {row['num_theme_words']}")
    print(f"\nboard[row][col]  (uppercase):")
    flat = row["board_flat"]
    for i in range(8):
        cols = [f"'{flat[i*6+j]}'" for j in range(6)]
        print(f"  board[{i}] = [{', '.join(cols)}]")

    # append to CSV if requested
    if output_path:
        existing = _load_existing(output_path)
        if row["date"] not in existing:
            first_write = not output_path.exists() or output_path.stat().st_size == 0
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                if first_write:
                    w.writeheader()
                w.writerow(row)
            print(f"\nSaved → {output_path}")


# use following funcs to validate the existing csv
def _load_existing(path: Path) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with open(path, encoding="utf-8") as f:
        return {row["date"] for row in csv.DictReader(f)}


def validate(path: Path):
    if not path.exists():
        print("No CSV found."); return
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV is empty."); return

    good  = [r for r in rows if r.get("board_flat") and len(r["board_flat"]) == 48]
    dates = sorted(r["date"] for r in rows)
    print(f"Total rows  : {len(rows)}")
    print(f"Valid boards: {len(good)}/{len(rows)}")
    print(f"Date range  : {dates[0]} → {dates[-1]}")

    if good:
        r    = good[0]
        flat = r["board_flat"]
        print(f"\nSample — {r['date']}  #{r['puzzle_id']}  clue: \"{r['clue']}\"")
        print(f"  spanagram  : {r['spanagram']}")
        print(f"  board[row][col]:")
        for i in range(8):
            cols = [f"'{flat[i*6+j]}'" for j in range(6)]
            print(f"    board[{i}] = [{', '.join(cols)}]")
    print()



def main():
    p = argparse.ArgumentParser(
        description="Scrape Strands boards from strandsgame.me",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--date",     help="Fetch a single date (YYYY-MM-DD)")
    p.add_argument("--start",    default=START_DATE.isoformat())
    p.add_argument("--end",      default=date.today().isoformat())
    p.add_argument("--output",   default=OUTPUT_CSV)
    p.add_argument("--delay",    type=float, default=0.6,
                   help="Seconds between requests (default 0.6)")
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()

    out = Path(args.output)

    if args.validate:
        validate(out); return

    if args.date:
        fetch_single(date.fromisoformat(args.date), out); return

    bulk_scrape(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        output_path=out,
        delay=args.delay,
    )

if __name__ == "__main__":
    main()