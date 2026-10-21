from scripts.extract_maintenance import attach_footnotes
from scripts.extract_maintenance_llm import (
    EngineInterval,
    MaintenancePlan,
    Task,
    filter_engines,
    plan_to_sentences,
)

NAFTA = "Motores Nafta 1.2 12V /1.2 Turbo 1.6 16V / 1.6 THP"
TURBO = "Motor Nafta 1.0 Turbo 200"
DIESEL = "Todos los motores Diesel"
ENGINES = ["1.6 16V", "1.0 Turbo 200"]


def interval_task(text: str, *intervals: tuple[list[str], str]) -> Task:
    return Task(
        text=text,
        schedule="interval",
        revisions=[],
        intervals=[EngineInterval(engines=e, interval=i) for e, i in intervals],
    )


def engine_plan(*tasks: Task) -> MaintenancePlan:
    return MaintenancePlan(
        header="MOTORES",
        revision_interval_km=None,
        engine_columns=[NAFTA, TURBO, DIESEL],
        tasks=list(tasks),
    )

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


def test_filter_engines_keeps_a_cell_that_spans_every_column_as_one_interval():
    plan = engine_plan(
        interval_task("Sustitución del aceite*", ([NAFTA, TURBO, DIESEL], "10.000 km o 1 año"))
    )
    assert plan_to_sentences(filter_engines(plan, ENGINES))[1] == (
        "Sustitución del aceite*: 10.000 km o 1 año."
    )


def test_filter_engines_drops_other_engines_and_names_the_kept_ones_as_configured():
    plan = engine_plan(
        interval_task(
            "Sustitución de la correa*",
            (["Motor 1.0 Turbo 200"], "120.000 km o 6 años"),
            (["Motores 1.2 / 1.6 16V / THP"], "80.000 km o 4 años"),
            (["Motores Diesel"], "120.000 km o 4 años"),
        )
    )
    assert plan_to_sentences(filter_engines(plan, ENGINES))[1] == (
        "Sustitución de la correa*: 120.000 km o 6 años (1.0 Turbo 200); "
        "80.000 km o 4 años (1.6 16V)."
    )


def test_filter_engines_merges_kept_engines_with_the_same_interval():
    plan = engine_plan(
        interval_task(
            "Control de la correa*",
            ([NAFTA], "20.000 km o 2 años"),
            ([TURBO], "20.000 km o 2 años"),
            ([DIESEL], "40.000 km o 4 años"),
        )
    )
    assert plan_to_sentences(filter_engines(plan, ENGINES))[1] == (
        "Control de la correa*: 20.000 km o 2 años."
    )


def test_filter_engines_drops_a_row_only_for_other_engines():
    plan = engine_plan(
        interval_task("Llenado aditivo FAP", ([DIESEL], "80.000 km o 4 años")),
        interval_task("Control de luces", ([NAFTA, TURBO, DIESEL], "10.000 km o 1 año")),
    )
    filtered = filter_engines(plan, ENGINES)
    assert [task.text for task in filtered.tasks] == ["Control de luces"]
    assert filtered.engine_columns == ENGINES


def test_filter_engines_keeps_intervals_without_engines_and_other_rows():
    plan = MaintenancePlan(
        header="REVISIONES Cada 10.000 km",
        revision_interval_km=10000,
        engine_columns=[],
        tasks=[
            Task(text="FRENOS", schedule="heading", revisions=[], intervals=[]),
            Task(text="Control de luces", schedule="all", revisions=[], intervals=[]),
            interval_task("Sustitución de bujías", ([], "cada 30.000 km")),
        ],
    )
    assert plan_to_sentences(filter_engines(plan, ENGINES)) == [
        "REVISIONES Cada 10.000 km",
        "FRENOS",
        "Control de luces: en todas las revisiones.",
        "Sustitución de bujías: cada 30.000 km.",
    ]


def test_attach_footnotes_resolves_notes_in_the_middle_of_the_rows():
    resolved = attach_footnotes(
        [
            "Bujías: 20.000 km (**); 60.000 km.",
            "(**) Motores 1.2 Nafta",
            "Sustitución de frenos (*): 2 años.",
            "(*) Mitad de los plazos.",
        ]
    )
    assert resolved == [
        "Bujías: 20.000 km (**); 60.000 km. (**) Motores 1.2 Nafta",
        "(**) Motores 1.2 Nafta",
        "Sustitución de frenos (*): 2 años. (*) Mitad de los plazos.",
        "(*) Mitad de los plazos.",
    ]
