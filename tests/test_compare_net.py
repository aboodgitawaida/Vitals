"""compare_net: per-recording rows, extra statistics, and keeping the hand-written header of summary.md."""

import compare_net as c


def _rec(name, errs, first=7.0):
    windows = [{"end": 15.0 + 3 * i, "ref": 70.0, "network": 70.0 + e, "confidence": 0.9, "hand-made": 70.0 + 2 * e}
               for i, e in enumerate(errs)]
    screen = [(0.0, None, 70.0, False), (first, 71.0, 70.0, False), (20.0, 72.0, 70.0, False)]
    return {"name": name, "brightness": 90.0, "windows": {"15.0": windows, "8.0": windows},
            "screen": {"network": screen, "hand-made": screen}}


def test_per_recording_rows():
    rows = c.per_recording([_rec("2_after", [1, 2, 9]), _rec("1_before", [0, 1, 2], first=9.5)])
    assert [r["recording"] for r in rows] == ["1_before", "2_after"]
    assert set(rows[0]) == set(c.PER_RECORDING_COLUMNS)
    assert rows[0]["network_MAE"] == "1.00" and rows[0]["network_first_number_s"] == "9.5"
    assert rows[1]["network_within5"] == "0.67" and rows[1]["handmade_MAE"] == "8.00"


def test_extra_stats_and_header(tmp_path):
    recs = [_rec("1_before", [0, 1, 2]), _rec("1_after", [3, 9, 1])]
    text = "\n".join(c.extra_stats(recs))
    assert "| recordings with 15 s MAE under 5 BPM | 2 / 2 |" in text
    assert "| at rest: 15 s MAE / within 5 BPM | 1.00 / 100% |" in text
    path = tmp_path / "summary.md"
    assert c.keep_header(str(path)) == ""
    path.write_text("# Title\n\nWhy this run.\n\n200 recordings, 100 people. 95% intervals resample people.\n| old |\n")
    assert c.keep_header(str(path)) == "# Title\n\nWhy this run.\n\n"
