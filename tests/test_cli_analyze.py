from ip_hack_detectors.cli import main


def test_analyze_reports_a_missing_run_as_an_error(tmp_path, capsys):
    run = tmp_path / "missing"
    args = ["analyze", "--run", str(run), "--config", "configs/mbpp-fake.yaml"]
    args += ["--selection", "results/dcal_selection.json"]
    args += ["--labels", str(tmp_path / "a.csv"), "--rater", "a"]
    args += ["--second-labels", str(tmp_path / "b.csv"), "--second-rater", "b"]
    args += ["--out", str(tmp_path / "out")]
    assert main(args) == 2
    assert "error:" in capsys.readouterr().err


def test_reference_judgments_reports_a_missing_run_as_an_error(tmp_path, capsys):
    run = tmp_path / "missing"
    assert (
        main(["reference-judgments", "--run", str(run), "--config", "configs/mbpp-fake.yaml"]) == 2
    )
    assert "error:" in capsys.readouterr().err
