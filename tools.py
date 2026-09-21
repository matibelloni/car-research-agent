import pint

CONVERT_UNITS_TOOL = {
    "name": "convert_units",
    "description": "Converts a numeric value between units (e.g. mph to km/h, mpg to l/100km, gal to l, hp to kw). Always use this instead of doing the math yourself when normalizing a value to the metric system.",
    "input_schema": {
        "type": "object",
        "properties": {
            "unit_from": {
                "type": "string",
                "description": "The unit to convert from",
            },
            "unit_to": {
                "type": "string",
                "description": "The unit to convert to",
            },
            "value": {
                "type": "number",
                "description": "The value to convert",
            },
        },
        "required": ["unit_from", "unit_to", "value"],
    },
}

ureg = pint.UnitRegistry()

# Short/lowercase forms the model tends to send that pint's case-sensitive
# registry doesn't resolve on its own (e.g. "kw" bare would otherwise fail,
# since only "kW" is defined).
ALIASES = {"f": "degF", "c": "degC", "kw": "kW"}


def convert_units(unit_from: str, unit_to: str, value: float) -> str:
    """Converts a value between two units and returns the result as text."""
    unit_from = ALIASES.get(unit_from.strip().lower(), unit_from.strip().lower())
    unit_to = ALIASES.get(unit_to.strip().lower(), unit_to.strip().lower())

    # mpg <-> l/100km is a reciprocal relationship, not something pint parses directly.
    if {unit_from, unit_to} == {"mpg", "l/100km"}:
        result = 235.215 / value
        return f"{value} {unit_from} = {result:.4g} {unit_to}"

    try:
        result = ureg.Quantity(value, unit_from).to(unit_to)
    except (pint.errors.UndefinedUnitError, pint.errors.DimensionalityError):
        raise ValueError(f"No conversion available from '{unit_from}' to '{unit_to}'")

    # Round before formatting: near-zero results (e.g. 32 degF -> 0 degC) can
    # carry floating-point noise like 5.684e-14 that .4g would print verbatim.
    magnitude = round(result.magnitude, 6)
    return f"{value} {unit_from} = {magnitude:.4g} {result.units:~P}"
