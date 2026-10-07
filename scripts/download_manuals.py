"""Download the owner's manuals listed in MANUALS into data/manuals/."""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

MANUALS_DIR = Path(__file__).resolve().parent.parent / "data" / "manuals"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


@dataclass(frozen=True)
class Manual:
    make: str
    model: str
    edition_year: int
    url: str

    @property
    def filename(self) -> str:
        name = f"{self.make}-{self.model}-{self.edition_year}.pdf"
        return name.lower().replace(" ", "-")


MANUALS = [
    Manual(
        make="Fiat",
        model="Cronos",
        edition_year=2021,
        url="https://servicios.fiat.com.ar/content/dam/fiat/argentina/manuales/cronos/60351316-Manual-Cronos-Esp2021.pdf",
    ),
]


def download(manual: Manual) -> Path:
    path = MANUALS_DIR / manual.filename
    if path.exists():
        return path
    request = urllib.request.Request(manual.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        path.write_bytes(response.read())
    return path


def main() -> None:
    MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    for manual in MANUALS:
        path = download(manual)
        size_mb = path.stat().st_size / 1_000_000
        print(f"{path.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
