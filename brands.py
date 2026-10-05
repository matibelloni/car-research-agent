"""Brand matching shared by the ingest scripts and the recalls tool."""

import re
import unicodedata

FCA_BRANDS = ["Fiat", "Jeep", "RAM", "Chrysler", "Dodge"]
STELLANTIS_BRANDS = FCA_BRANDS + ["Peugeot", "Citroën", "DS"]

# Matched as whole words against the normalized company name.
# A publisher can map to several brands on purpose: retrieval should
# over-include, and the agent discards rows whose product doesn't match.
BRAND_KEYWORDS: list[tuple[str, list[str]]] = [
    ("audi", ["Audi"]),
    ("bmw", ["BMW"]),
    ("chery", ["Chery"]),
    ("chevrolet", ["Chevrolet"]),
    ("coopermex", ["Cooper"]),  # tire recall, not a vehicle maker
    ("ds", ["DS"]),  # two-letter keyword: \b prevents matches inside words like "SKUS"
    ("general motors", ["Chevrolet"]),
    ("citroen", ["Citroën"]),
    ("cooper", ["Cooper"]),
    ("peugeot", ["Peugeot"]),
    ("fca", FCA_BRANDS),
    ("stellantis", STELLANTIS_BRANDS),
    ("fiat", ["Fiat"]),
    ("jeep", ["Jeep"]),
    ("ram", ["RAM"]),
    ("chrysler", ["Chrysler"]),
    ("dodge", ["Dodge"]),
    ("eximar", ["Volvo", "Jaguar", "Land Rover"]),  # importer of all three
    ("ford", ["Ford"]),
    ("hino", ["Hino"]),
    ("honda", ["Honda"]),
    ("jaguar", ["Jaguar"]),
    ("kawasaki", ["Kawasaki"]),
    ("kia", ["Kia"]),
    ("land rover", ["Land Rover"]),
    ("lexus", ["Lexus"]),
    ("toyota", ["Toyota", "Lexus"]),  # Toyota Argentina also publishes Lexus recalls
    ("mercedes", ["Mercedes-Benz"]),
    ("mitsubishi", ["Mitsubishi"]),
    ("mitsu motors", ["Mitsubishi"]),  # Mitsubishi distributor
    ("nissan", ["Nissan"]),
    ("renault", ["Renault"]),
    ("suzuki", ["Suzuki"]),
    ("volkswagen", ["Volkswagen", "Audi", "Seat"]),
    ("vw", ["Volkswagen", "Audi", "Seat"]),
    ("volvo", ["Volvo"]),
    ("yamaha", ["Yamaha"]),
    ("yahama", ["Yamaha"]),  # typo present in the source sheet
    ("seat", ["Seat"]),
]
AVAILABLE_BRANDS = sorted({b for _, brands in BRAND_KEYWORDS for b in brands})


def normalize(text: str) -> str:
    """Trim, lowercase and strip accents, so ' Vehículos' == 'vehiculos'."""
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def brands_for(name: str) -> list[str]:
    """Canonical brands mentioned in a publisher or brand name."""
    text = normalize(name)
    brands: set[str] = set()
    for keyword, keyword_brands in BRAND_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            brands.update(keyword_brands)
    return sorted(brands)
