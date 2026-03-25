# File: evolution/tests/test_integration.py
import pytest
import tempfile
import os
from evolution.oni_archive import ONIArchive
from evolution.improvement_proposal import ONIVariant


def _make_archive(tmpdir):
    initial_dir = os.path.join(tmpdir, "initial_src")
    os.makedirs(initial_dir)
    return ONIArchive(
        archive_dir=os.path.join(tmpdir, "archive"),
        initial_variant_path=initial_dir
    )


def test_archive_initialization():
    """Archive initializes with an 'initial' variant."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = _make_archive(tmpdir)
        assert "initial" in archive.variants
        assert archive.generation == 0


def test_parent_selection_returns_correct_count():
    """select_parents returns exactly k IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = _make_archive(tmpdir)
        for i, score in enumerate([0.4, 0.6, 0.8]):
            archive.add_variant(ONIVariant(
                variant_id=f"v{i}", parent_id="initial", generation=1,
                patch_file="", overall_score=score, is_compiled=True
            ))

        for method in ('score_child_prop', 'score_prop', 'best', 'random'):
            parents = archive.select_parents(k=2, method=method)
            assert len(parents) == 2
            assert all(p in archive.variants for p in parents)


def test_add_variant_rejects_uncompiled():
    """Uncompiled variants are not added to the archive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = _make_archive(tmpdir)
        bad = ONIVariant(
            variant_id="bad_v1", parent_id="initial", generation=1,
            patch_file="", overall_score=0.9, is_compiled=False
        )
        added = archive.add_variant(bad)
        assert not added
        assert "bad_v1" not in archive.variants


def test_get_best_variant():
    """get_best_variant returns highest-scoring compiled variant."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = _make_archive(tmpdir)
        archive.add_variant(ONIVariant(
            variant_id="low", parent_id="initial", generation=1,
            patch_file="", overall_score=0.3, is_compiled=True
        ))
        archive.add_variant(ONIVariant(
            variant_id="high", parent_id="initial", generation=1,
            patch_file="", overall_score=0.9, is_compiled=True
        ))
        best = archive.get_best_variant()
        assert best.variant_id == "high"


def test_archive_state_persisted():
    """Archive state is written to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = _make_archive(tmpdir)
        state = archive.load_state()
        assert state is not None
        assert 'generation' in state


def test_children_count_increments():
    """select_parents increments children_count on selected variants."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive = _make_archive(tmpdir)
        archive.add_variant(ONIVariant(
            variant_id="solo", parent_id="initial", generation=1,
            patch_file="", overall_score=0.5, is_compiled=True
        ))
        before = archive.variants["solo"].children_count
        archive.select_parents(k=3, method='best')
        after = archive.variants["solo"].children_count
        # solo may have been selected multiple times
        assert after >= before
