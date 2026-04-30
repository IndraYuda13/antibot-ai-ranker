from claimcoin_antibot_ai_ranker.paths import PROJECTS_ROOT, SourcePaths


def test_default_source_paths_point_to_sibling_projects():
    paths = SourcePaths()
    assert paths.claimcoin_root == PROJECTS_ROOT / "claimcoin-autoclaim"
    assert paths.solver_root == PROJECTS_ROOT / "antibot-image-solver"
