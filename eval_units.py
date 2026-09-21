from tools import convert_units

# (unit_from, unit_to, value, expected_result_substring)
cases = [
    ("mph", "km/h", 60, "96.56 km/h"),
    ("mi", "km", 10, "16.09 km"),
    ("gal", "l", 1, "3.785 l"),
    ("hp", "kw", 100, "74.57 kW"),
    ("lb", "kg", 220, "99.79 kg"),
    ("in", "cm", 5, "12.7 cm"),
    ("psi", "bar", 30, "2.068 bar"),
    ("f", "c", 32, "0 °C"),
    ("mpg", "l/100km", 25, "9.409 l/100km"),
    ("ft", "m", 10, "3.048 m"),
    ("feet", "meters", 10, "3.048 m"),
]

successes = 0
for unit_from, unit_to, value, expected in cases:
    result = convert_units(unit_from, unit_to, value)
    ok = expected in result
    successes += ok

    print(f"{'✅' if ok else '❌'} {value} {unit_from} -> {unit_to}: {result}")
    if not ok:
        print(f"   expected to contain: {expected}")

print(f"Precision: successes/len(cases): {successes/len(cases):.2f}")

# Unsupported pair must raise, so the agent can signal is_error to the model.
try:
    convert_units("foo", "bar", 1)
    print("❌ expected ValueError for unsupported units")
except ValueError:
    print("✅ raises ValueError for unsupported units")
