from scripts.extract_maintenance import attach_footnotes

TASKS = [
    "REVISIONES Cada 10.000 km",
    "Control visual de las correas (**): en las revisiones 2ª (20.000 km).",
    "Sustitución de las correas (*): cada 48 meses.",
    "Control de luces: en todas las revisiones.",
]
NOTES = [
    "(*) Mitad de los plazos en uso severo.",
    "(**) Control cada 10.000 km en caminos polvorientos.",
]


def test_attach_footnotes_matches_whole_markers():
    resolved = attach_footnotes(TASKS + [" ".join(NOTES)])
    assert resolved[1].endswith(": en las revisiones 2ª (20.000 km). " + NOTES[1])
    assert resolved[2] == "Sustitución de las correas (*): cada 48 meses. " + NOTES[0]


def test_attach_footnotes_leaves_unmarked_rows_and_keeps_notes():
    one_cell = " ".join(NOTES)
    resolved = attach_footnotes(TASKS + [one_cell])
    assert resolved[0] == TASKS[0]
    assert resolved[3] == TASKS[3]
    assert resolved[-1] == one_cell


def test_attach_footnotes_one_row_per_note_as_the_llm_returns_them():
    assert attach_footnotes(TASKS + NOTES)[:4] == attach_footnotes(
        TASKS + [" ".join(NOTES)]
    )[:4]
