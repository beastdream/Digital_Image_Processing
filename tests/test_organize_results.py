from pathlib import Path

from src.utils.organize_results import initialize_results, main, migrate_results, preview_migration


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_migrates_old_preprocessing_to_canonical_destination(tmp_path: Path) -> None:
    _write(tmp_path / "results/preprocessing_visualizations/old.jpg", "old")
    actions = migrate_results(tmp_path)
    assert actions
    assert (tmp_path / "results/preprocessing/visualizations/old.jpg").read_text() == "old"
    assert not (tmp_path / "results/preprocessing_visualizations").exists()
    assert migrate_results(tmp_path) == []


def test_preprocessing_collision_goes_to_legacy_without_overwrite(tmp_path: Path) -> None:
    _write(tmp_path / "results/preprocessing/visualizations/current.jpg", "current")
    _write(tmp_path / "results/preprocessing_visualizations/current.jpg", "old")
    migrate_results(tmp_path)
    assert (tmp_path / "results/preprocessing/visualizations/current.jpg").read_text() == "current"
    assert (tmp_path / "results/legacy/preprocessing_visualizations/current.jpg").read_text() == "old"


def test_old_direct_predictions_move_to_legacy_not_latest(tmp_path: Path) -> None:
    _write(tmp_path / "results/predictions/pred_one.jpg", "one")
    _write(tmp_path / "results/predictions/pred_two.png", "two")
    migrate_results(tmp_path)
    assert (tmp_path / "results/legacy/predictions_previous/pred_one.jpg").is_file()
    assert (tmp_path / "results/legacy/predictions_previous/pred_two.png").is_file()
    assert list((tmp_path / "results/predictions/latest").iterdir()) == []
    assert migrate_results(tmp_path) == []


def test_old_yolo_tree_moves_wholly_to_legacy_and_preserves_collisions(tmp_path: Path) -> None:
    _write(tmp_path / "results/legacy/yolo/baseline/metrics.json", "existing")
    _write(tmp_path / "results/yolo/baseline/metrics.json", "old")
    _write(tmp_path / "results/yolo/debug/report.json", "debug")
    actions = migrate_results(tmp_path)
    legacy = tmp_path / "results/legacy/yolo"
    assert (legacy / "baseline/metrics.json").read_text() == "existing"
    assert (legacy / "baseline/metrics.legacy_1.json").read_text() == "old"
    assert (legacy / "debug/report.json").read_text() == "debug"
    assert any(action.startswith("ARCHIVE ") for action in actions)
    assert not (tmp_path / "results/yolo").exists()
    assert migrate_results(tmp_path) == []


def test_migration_never_touches_protected_project_trees(tmp_path: Path) -> None:
    protected = [
        tmp_path / "experiments/yolo/A_raw/weights/best.pt",
        tmp_path / "data/raw/source.jpg",
        tmp_path / "data/processed/dataset.yaml",
        tmp_path / "configs/config.yaml",
        tmp_path / "src/module.py",
        tmp_path / "tests/test_module.py",
    ]
    for path in protected:
        _write(path, f"protected:{path.name}")
    before = {path: path.read_bytes() for path in protected}
    _write(tmp_path / "results/yolo/unknown/result.json", "legacy")
    migrate_results(tmp_path)
    assert {path: path.read_bytes() for path in protected} == before


def test_moves_only_known_legacy_analysis_files_and_keeps_dataset_reports(tmp_path: Path) -> None:
    known = tmp_path / "data/reports/class_imbalance_and_bbox_analysis.json"
    unrelated = tmp_path / "data/reports/unrelated.json"
    dataset_report = tmp_path / "data/processed/road_damage_detection/reports/invalid_annotations.json"
    _write(known, "analysis")
    _write(unrelated, "unrelated")
    _write(dataset_report, "dataset integrity")
    migrate_results(tmp_path)
    assert (tmp_path / "results/analysis/class_balance/class_imbalance_and_bbox_analysis.json").read_text() == "analysis"
    assert unrelated.read_text() == "unrelated"
    assert dataset_report.read_text() == "dataset integrity"


def test_existing_class_balance_artifact_is_not_overwritten(tmp_path: Path) -> None:
    old = tmp_path / "data/reports/bbox_size_statistics.csv"
    current = tmp_path / "results/analysis/class_balance/bbox_size_statistics.csv"
    _write(old, "old")
    _write(current, "current")
    migrate_results(tmp_path)
    assert current.read_text() == "current"
    assert (tmp_path / "results/legacy/class_balance_previous/bbox_size_statistics.csv").read_text() == "old"


def test_clean_layout_prints_required_idempotent_message(tmp_path: Path, capsys) -> None:
    (tmp_path / "results").mkdir()
    assert main(["--migrate"], root=tmp_path) == 0
    assert capsys.readouterr().out.strip() == "Results layout is already organized. No migration required."


def test_migration_cli_logs_move_actions_clearly(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "results/yolo/baseline/metrics.json", "metric")
    assert main(["--migrate"], root=tmp_path) == 0
    assert "- MOVE results/yolo -> results/legacy/yolo" in capsys.readouterr().out


def _snapshot(root: Path) -> list[tuple[str, str, bytes | None]]:
    return [
        (path.relative_to(root).as_posix(), "dir" if path.is_dir() else "file",
         None if path.is_dir() else path.read_bytes())
        for path in sorted(root.rglob("*"))
    ]


def test_dry_run_prints_plan_without_any_filesystem_change(tmp_path: Path, capsys) -> None:
    _write(tmp_path / "results/yolo/baseline/metrics.json", "metric")
    _write(tmp_path / "results/preprocessing_visualizations/sample.jpg", "sample")
    _write(tmp_path / "results/predictions/pred_old.jpg", "prediction")
    before = _snapshot(tmp_path)

    assert main(["--migrate", "--dry-run"], root=tmp_path) == 0
    output = capsys.readouterr().out
    assert "MOVE results/yolo -> results/legacy/yolo" in output
    assert "MOVE results/preprocessing_visualizations -> results/preprocessing/visualizations" in output
    assert "MOVE results/predictions/pred_old.jpg -> results/legacy/predictions_previous/pred_old.jpg" in output
    assert "CREATE results/predictions/latest" in output
    assert _snapshot(tmp_path) == before
    assert preview_migration(tmp_path)


def test_init_creates_complete_structure_without_deleting_content(tmp_path: Path, capsys) -> None:
    sentinel = tmp_path / "results/experiments/manual_review.md"
    _write(sentinel, "preserve")
    actions = initialize_results(tmp_path)
    assert actions
    expected = (
        "results/predictions/latest",
        "results/evaluations",
        "results/experiments",
        "results/preprocessing/visualizations",
        "results/analysis/class_balance",
        "results/legacy",
    )
    assert all((tmp_path / path).is_dir() for path in expected)
    assert sentinel.read_text() == "preserve"
    assert main(["--init"], root=tmp_path) == 0
    assert capsys.readouterr().out.strip() == "Results directory structure is already initialized."
