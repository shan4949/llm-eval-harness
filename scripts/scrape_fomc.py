import httpx
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import date
import time

BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
OUTPUT_DIR = Path(__file__).parent.parent / "sut" / "data"

def get_latest_downloaded_date() -> tuple[int, int] | None:
    """Return (year, month) of the most recent file in sut/data/, or None if empty."""
    files = sorted(OUTPUT_DIR.glob("fomc_*.txt"))
    if not files:
        return None
    # filename looks like fomc_2024_03.txt
    last = files[-1].stem          # "fomc_2024_03"
    parts = last.split("_")        # ["fomc", "2024", "03"]
    return int(parts[1]), int(parts[2])   # (2024, 3)

def get_statement_links() -> list[str]:
    latest = get_latest_downloaded_date()
    response = httpx.get(CALENDAR_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "monetary" in href and href.endswith("a.htm"):
            year = int(href[href.index("monetary") + 8:][:4])
            month = int(href[href.index("monetary") + 12:][:2])
            # skip anything we already have
            if latest and (year, month) <= latest:
                continue
            links.append(BASE_URL + href)
    return links

def scrape_statement(url: str) -> str:
    """Download one statement page and return clean text."""
    response = httpx.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # the content is inside this div on the Fed site
    content = soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
    if not content:
        return ""

    # hint: content.get_text(separator="\n") gives you the raw text
    # hint: strip each line, filter out empty lines
    lines = [line.strip() for line in content.get_text(separator="\n").splitlines() if line.strip()]
    return "\n".join(lines)

def url_to_filename(url: str) -> str:
    """Turn the URL into a readable filename."""
    # url looks like: .../monetary20240320a.htm
    # hint: url.split("/")[-1] gives you "monetary20240320a.htm"
    # extract year=2024, month=03 → return "fomc_2024_03.txt"
    part = url.split("/")[-1]          # "monetary20240320a.htm"
    year = part[8:12]          # "2024"
    month = part[12:14]         # "03"
    return f"fomc_{year}_{month}.txt"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    links = get_statement_links()
    print(f"Found {len(links)} statements")

    for url in links:
        filename = url_to_filename(url)
        output_path = OUTPUT_DIR / filename

        if output_path.exists():
            print(f"  skipping {filename} (already downloaded)")
            continue

        print(f"  downloading {filename}...")
        text = scrape_statement(url)

        if text:
            output_path.write_text(text, encoding="utf-8")

        time.sleep(1)  # be polite to their server

if __name__ == "__main__":
    main()
