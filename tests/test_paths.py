from antibot_ai_ranker.paths import PROJECTS_ROOT, SourcePaths


def test_default_source_paths_point_to_current_local_dataset_adapter():
    paths = SourcePaths()
    assert paths.source_root == PROJECTS_ROOT / "claimcoin-autoclaim"
    assert paths.solver_root == PROJECTS_ROOT / "antibot-image-solver"
    assert paths.db_name == "claimcoin.sqlite3"
    assert paths.case_prefix == "claimcoin"
