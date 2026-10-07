"""
Tests for the Netflix Recommendation System core logic (app.py).

These tests exercise the same public functions/classes the Streamlit app
uses — app.load_catalogue() and app.NetflixRecommender.recommend() —
against the repo's real catalogue (data.csv, 8,790 rows).

Run from the repo root:  pytest -q
"""

import sys
from pathlib import Path

import pytest

# app.py lives in the repo root, one level above this tests/ folder.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import app  # noqa: E402


@pytest.fixture(scope="module")
def catalogue():
    """The catalogue exactly as the app loads it (default data.csv path)."""
    return app.load_catalogue()


@pytest.fixture(scope="module")
def recommender(catalogue):
    """One shared recommender — building the TF-IDF matrix takes a moment."""
    return app.NetflixRecommender(catalogue)


def test_dataset_loads_with_8787_unique_titles(catalogue):
    # data.csv has 8,790 rows; 3 titles are duplicated, and load_catalogue()
    # drops duplicates keeping the first entry.
    assert len(catalogue) == 8787
    assert catalogue["title"].nunique() == 8787


def test_recommend_midnight_mass_top10_includes_hill_house(recommender):
    recs = recommender.recommend("Midnight Mass")  # default top_n=10
    assert len(recs) == 10
    top5 = recs["title"].head(5).tolist()
    assert "The Haunting of Hill House" in top5


def test_recommendations_exclude_the_query_title(recommender):
    recs = recommender.recommend("Midnight Mass")
    assert "Midnight Mass" not in recs["title"].tolist()


def test_match_scores_between_0_and_100_sorted_descending(recommender):
    recs = recommender.recommend("Midnight Mass")
    # The UI presents similarity (0..1) as a percentage match (0..100).
    match = (recs["similarity"] * 100).tolist()
    assert all(0 <= m <= 100 for m in match)
    assert match == sorted(match, reverse=True)
