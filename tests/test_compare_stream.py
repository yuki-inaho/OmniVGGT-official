import json

import pytest


def _candidate(name, kv_bytes, s0, s1):
    return {"name": name, "kv_bytes": kv_bytes, "sessions": {"s0": s0, "s1": s1}}


def _result(s0, s1, kv_bytes=None, split="smoke", precision="fp32", mode="stream"):
    """An eval_stream result with one primary value per window."""
    windows = []
    for session, values in (("s0", s0), ("s1", s1)):
        for index, value in enumerate(values):
            efficiency = {"per_frame_ms": 30.0 + index}
            if kv_bytes is not None:
                efficiency["kv_bytes_max"] = kv_bytes - index
            metrics = {"ate_g1_rmse_mm": value, "ate_sim3_rmse_mm": value / 2, "abs_rel_s1": 0.02}
            windows.append(
                {
                    "spec": f"{session}:{950 + index}-981",
                    "session": session,
                    "conditions": {"depth": {"metrics": metrics, "efficiency": efficiency}},
                }
            )
    return {
        "split": split,
        "roots": ["root_a", "root_b"],
        "resolution_wh": [392, 294],
        "provenance": {"mode": mode, "policy": "full", "precision": {"name": precision}},
        "windows": windows,
    }


def test_session_means_average_the_windows_of_each_session():
    import compare_stream

    result = _result([10.0, 20.0, 30.0], [40.0, 50.0, 60.0], kv_bytes=1000)
    assert compare_stream.session_means(result) == {"s0": 20.0, "s1": 50.0}
    assert compare_stream.kv_bytes(result) == 1000


def test_the_smallest_kv_candidate_is_the_reference():
    import compare_stream

    candidates = [_candidate("w8", 900, 50, 50), _candidate("w1", 200, 100, 100), _candidate("w4", 500, 99, 99)]
    table = compare_stream.select(candidates)
    assert table["reference"] == "w1" and [row["name"] for row in table["rows"]] == ["w1", "w4", "w8"]
    tied = [_candidate("query", 300, 100, 100), _candidate("random", 300, 100, 100)]
    assert compare_stream.select(tied)["reference"] == "query"  # equal KV bytes keep the listed order


def test_a_candidate_needs_five_percent_and_the_same_direction_in_both_sessions():
    import compare_stream

    reference = _candidate("ref", 100, 100.0, 100.0)
    exactly_five = _candidate("five", 200, 94.0, 96.0)  # mean 95: 5% better, both sessions better
    table = compare_stream.select([reference, exactly_five])
    assert table["selected"] == "five"
    assert table["rows"][1]["improvement"] == pytest.approx(0.05) and table["rows"][1]["qualifies"]
    below = _candidate("four", 200, 96.0, 96.0)
    assert compare_stream.select([reference, below])["selected"] == "ref"


def test_an_improvement_in_one_session_only_is_not_selected():
    import compare_stream

    one_sided = _candidate("s0-only", 200, 70.0, 101.0)  # mean 85.5 is 14.5% better, but s1 is worse
    table = compare_stream.select([_candidate("ref", 100, 100.0, 100.0), one_sided])
    assert table["selected"] == "ref"
    assert table["rows"][1]["improvement"] == pytest.approx(0.145) and not table["rows"][1]["same_direction"]


def test_the_best_mean_wins_among_qualifying_candidates():
    import compare_stream

    candidates = [
        _candidate("ref", 100, 100.0, 100.0),
        _candidate("a", 200, 90.0, 90.0),
        _candidate("b", 300, 80.0, 98.0),  # mean 89: best
        _candidate("c", 400, 60.0, 101.0),  # mean 80.5 but s1 worse
    ]
    table = compare_stream.select(candidates)
    assert table["selected"] == "b"
    assert [row["qualifies"] for row in table["rows"]] == [False, True, True, False]


def test_candidates_must_cover_the_same_sessions():
    import compare_stream

    odd = {"name": "odd", "kv_bytes": 200, "sessions": {"s0": 1.0}}
    with pytest.raises(ValueError, match="sessions"):
        compare_stream.select([_candidate("ref", 100, 1.0, 1.0), odd])


@pytest.mark.parametrize(
    "a,b,differs,better",
    [
        ((90.0, 95.0), (100.0, 100.0), True, "a"),  # 7.5% lower in the mean, lower in both sessions
        ((105.0, 106.0), (100.0, 100.0), True, "b"),  # 5.5% higher, higher in both
        ((97.0, 97.0), (100.0, 100.0), False, None),  # 3%
        ((80.0, 110.0), (100.0, 100.0), False, None),  # 5% but opposite directions
        ((100.0, 90.0), (100.0, 100.0), False, None),  # s0 unchanged
    ],
)
def test_difference_rule(a, b, differs, better):
    import compare_stream

    verdict = compare_stream.differs({"s0": a[0], "s1": a[1]}, {"s0": b[0], "s1": b[1]})
    assert verdict["differs"] is differs and verdict["better"] == better
    assert verdict["relative_difference"] == pytest.approx((sum(a) - sum(b)) / sum(b))


def test_selection_is_made_on_the_smoke_split_only():
    import compare_stream

    results = {"w1": _result([10.0], [10.0], kv_bytes=100), "w4": _result([9.0], [9.0], kv_bytes=400)}
    assert compare_stream.selection_table(results, "CA")["selected"] == "w4"
    results["w4"] = _result([9.0], [9.0], kv_bytes=400, split="val")
    with pytest.raises(ValueError, match="split"):
        compare_stream.selection_table(results, "CA")
    with pytest.raises(ValueError, match="KV"):
        compare_stream.selection_table({"w1": _result([10.0], [10.0], kv_bytes=100), "r0": _result([9.0], [9.0])}, "CA")


def test_results_must_share_the_protocol():
    import compare_stream

    base = _result([10.0, 11.0], [10.0, 11.0], kv_bytes=100)
    with pytest.raises(ValueError, match="windows"):
        compare_stream.check_protocol({"a": base, "b": _result([10.0], [10.0], kv_bytes=100)})
    with pytest.raises(ValueError, match="precision"):
        compare_stream.check_protocol({"a": base, "b": _result([10.0, 11.0], [10.0, 11.0], 100, precision="bf16")})
    compare_stream.check_protocol({"a": base, "b": _result([9.0, 9.0], [9.0, 9.0], mode="bidir_f0")})


def test_confirmation_table_and_comparisons():
    import compare_stream

    results = {
        "R3": _result([90.0, 90.0], [95.0, 95.0], kv_bytes=5000, split="val"),
        "R3b": _result([100.0, 100.0], [100.0, 100.0], split="val", mode="bidir_f0"),
    }
    table = compare_stream.confirmation_table(results, [("R3", "R3b")])
    rows = {row["name"]: row for row in table["rows"]}
    assert rows["R3"]["sessions"] == {"s0": 90.0, "s1": 95.0} and rows["R3"]["mean"] == 92.5
    assert rows["R3"]["kv_bytes"] == 5000 and "kv_bytes" not in rows["R3b"]
    assert rows["R3b"]["mode"] == "bidir_f0" and rows["R3"]["per_frame_ms"] == pytest.approx(30.5)
    (comparison,) = table["comparisons"]
    assert (
        comparison["a"] == "R3" and comparison["b"] == "R3b" and comparison["differs"] and comparison["better"] == "a"
    )
    with pytest.raises(ValueError, match="R9"):
        compare_stream.confirmation_table(results, [("R3", "R9")])


def _write(path, result):
    path.write_text(json.dumps(result))
    return f"{path.stem}={path}"


def test_cli_writes_markdown_and_json(tmp_path):
    import compare_stream

    w1 = _write(tmp_path / "w1.json", _result([10.0, 10.0], [10.0, 10.0], kv_bytes=100))
    w4 = _write(tmp_path / "w4.json", _result([9.0, 9.0], [9.4, 9.4], kv_bytes=400))
    prefix = tmp_path / "out" / "sel_CA"
    assert compare_stream.main(["--stage", "CA", "--results", w1, w4, "--output", str(prefix)]) == 0
    table = json.loads(prefix.with_suffix(".json").read_text())
    assert table["stage"] == "CA" and table["reference"] == "w1" and table["selected"] == "w4"
    assert set(table["inputs"]) == {"w1", "w4"} and len(table["inputs"]["w1"]["sha256"]) == 64
    markdown = prefix.with_suffix(".md").read_text()
    assert "| w4 |" in markdown and "selected: w4" in markdown
    with pytest.raises(FileExistsError):
        compare_stream.main(["--stage", "CA", "--results", w1, w4, "--output", str(prefix)])

    r3 = _write(tmp_path / "R3.json", _result([9.0, 9.0], [9.0, 9.0], kv_bytes=100, split="val"))
    r3b = _write(tmp_path / "R3b.json", _result([10.0, 10.0], [10.0, 10.0], split="val", mode="bidir_f0"))
    confirm = tmp_path / "out" / "confirm"
    argv = ["--confirm", "--results", r3, r3b, "--compare", "R3:R3b", "--output", str(confirm)]
    assert compare_stream.main(argv) == 0
    assert json.loads(confirm.with_suffix(".json").read_text())["comparisons"][0]["differs"] is True
    assert "| R3 | R3b |" in confirm.with_suffix(".md").read_text()
    for bad in (
        ["--results", w1, "--output", str(tmp_path / "x")],  # neither --stage nor --confirm
        ["--stage", "CA", "--confirm", "--results", w1, "--output", str(tmp_path / "x")],
        ["--stage", "CA", "--results", "w1.json", "--output", str(tmp_path / "x")],  # not NAME=PATH
    ):
        with pytest.raises((SystemExit, ValueError)):
            compare_stream.main(bad)
