"""
🎬 Netflix Recommendation System — Streamlit live demo
======================================================
A content-based recommender built with TF-IDF + cosine similarity over the
8,790-title Netflix catalogue. This is the live-demo version of an end-to-end ML project
(recommendation_system.py): every
title is turned into one short "profile" string built from its genres
(listed_in), country, director, type and rating; TF-IDF vectorises those
profiles and cosine similarity finds the closest titles.

Author: Muhammad Abdul Rafay — Machine Learning Intern
GitHub: https://github.com/mabdulrafay7zip
"""

import html
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data.csv"

DISPLAY_COLS = ["title", "type", "director", "country",
                "release_year", "rating", "duration", "listed_in"]


# --------------------------------------------------------------------------
# Core recommender (same approach as the original Task 1 script)
# --------------------------------------------------------------------------

def load_catalogue(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the Netflix catalogue and fill the gaps we care about."""
    df = pd.read_csv(path)
    for col in ["director", "country", "rating", "listed_in"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    # A duplicated title would confuse the lookup, keep the first entry.
    df = df.drop_duplicates(subset="title").reset_index(drop=True)
    return df


def _squash(text: str) -> str:
    """Remove spaces so multi-word names/genres stay a single TF-IDF token."""
    return str(text).replace(" ", "").replace(",", " ")


def build_profiles(df: pd.DataFrame) -> pd.Series:
    """Combine the descriptive columns into a single text profile per title."""
    profiles = (
        df["listed_in"].apply(_squash) + " "
        + df["country"].apply(_squash) + " "
        + df["director"].apply(_squash) + " "
        + df["type"].str.replace(" ", "") + " "
        + df["rating"]
    )
    # If a richer catalogue (with cast / description columns) is ever used,
    # fold those in as well — the shipped data.csv does not contain them.
    if "cast" in df.columns:
        profiles = profiles + " " + df["cast"].fillna("").apply(_squash)
    if "description" in df.columns:
        profiles = profiles + " " + df["description"].fillna("")
    return profiles


class NetflixRecommender:
    """Simple content-based recommender over the Netflix catalogue."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(build_profiles(df))
        # Title -> row position lookup (case-insensitive).
        self.index_of = {t.lower(): i for i, t in enumerate(df["title"])}

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """Return the top_n titles most similar to `title`."""
        key = title.strip().lower()
        if key not in self.index_of:
            raise ValueError(f"'{title}' was not found in the catalogue.")
        pos = self.index_of[key]
        scores = cosine_similarity(self.matrix[pos], self.matrix).ravel()
        # Rank everything, then drop the title itself (score 1.0).
        ranked = scores.argsort()[::-1]
        ranked = [i for i in ranked if i != pos][:top_n]
        cols = [c for c in DISPLAY_COLS if c in self.df.columns]
        out = self.df.iloc[ranked][cols].copy()
        out.insert(1, "similarity", scores[ranked].round(4))
        return out.reset_index(drop=True)


def describe(row: pd.Series) -> str:
    """One-line descriptive snippet for a catalogue row."""
    bits = []
    if str(row.get("director", "Unknown")) != "Unknown":
        bits.append(f"Directed by {row['director']}")
    if str(row.get("country", "Unknown")) != "Unknown":
        bits.append(f"from {row['country']}")
    snippet = " ".join(bits)
    genres = str(row.get("listed_in", ""))
    if genres and genres != "Unknown":
        snippet = f"{snippet} — {genres}." if snippet else f"{genres}."
    return snippet or "No details available."


# --------------------------------------------------------------------------
# Streamlit caches (data + TF-IDF model are built exactly once)
# --------------------------------------------------------------------------

@st.cache_data
def get_catalogue() -> pd.DataFrame:
    return load_catalogue()


@st.cache_resource
def get_recommender() -> NetflixRecommender:
    return NetflixRecommender(get_catalogue())


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

CSS = """
<style>
  .hero {
    background: linear-gradient(120deg, #141414 55%, #7a0c12 85%, #e50914 130%);
    border-radius: 14px; padding: 28px 32px; margin-bottom: 20px;
    border: 1px solid #2b2b2b;
  }
  .hero h1 { color: #ffffff; margin: 0 0 6px 0; font-size: 2.1rem; }
  .hero p  { color: #d7d7d7; margin: 0; font-size: 1.02rem; }
  .hero .brand { color: #e50914; font-weight: 800; letter-spacing: 1px; }
  .picked {
    background: #1b1b1b; border-left: 5px solid #e50914; border-radius: 10px;
    padding: 14px 18px; margin: 10px 0 18px 0; color: #eeeeee;
  }
  .rec-card {
    background: #1b1b1b; border: 1px solid #2e2e2e; border-radius: 12px;
    padding: 16px 18px; margin-bottom: 14px; height: 100%;
  }
  .rec-rank { color: #e50914; font-weight: 800; font-size: 0.85rem;
              letter-spacing: 1px; text-transform: uppercase; }
  .rec-title { color: #ffffff; font-weight: 700; font-size: 1.13rem;
               margin: 3px 0 6px 0; line-height: 1.3; }
  .rec-meta { color: #c9c9c9; font-size: 0.92rem; margin-bottom: 6px; }
  .rec-genres { color: #9fd3ff; font-size: 0.9rem; margin-bottom: 6px; }
  .rec-snippet { color: #a8a8a8; font-size: 0.9rem; margin-bottom: 10px; }
  .rec-match { color: #46d369; font-weight: 700; font-size: 0.95rem; }
  .footer { color: #8c8c8c; text-align: center; font-size: 0.9rem;
            margin-top: 28px; }
  .footer a { color: #e50914; text-decoration: none; }
</style>
"""


def render_card(rank: int, row: pd.Series) -> None:
    icon = "📺" if row["type"] == "TV Show" else "🎞️"
    meta = (f"{icon} {html.escape(str(row['type']))} · "
            f"{int(row['release_year'])} · {html.escape(str(row['rating']))} · "
            f"{html.escape(str(row['duration']))}")
    st.markdown(
        f"""
        <div class="rec-card">
          <div class="rec-rank">#{rank} recommendation</div>
          <div class="rec-title">{html.escape(str(row['title']))}</div>
          <div class="rec-meta">{meta}</div>
          <div class="rec-genres">{html.escape(str(row['listed_in']))}</div>
          <div class="rec-snippet">{html.escape(describe(row))}</div>
          <div class="rec-match">🍿 {row['similarity'] * 100:.1f}% match</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Netflix Recommendation System",
                       page_icon="🎬", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    df = get_catalogue()
    recommender = get_recommender()

    st.markdown(
        f"""
        <div class="hero">
          <div class="brand">NETFLIX&nbsp;RECOMMENDER</div>
          <h1>🎬 Netflix Recommendation System</h1>
          <p>Pick a title you love and get instant, content-based recommendations
             from a catalogue of <b>{len(df):,}</b> Netflix movies &amp; TV shows —
             powered by <b>TF-IDF + cosine similarity</b> (scikit-learn).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pick, col_n = st.columns([3, 1])
    with col_pick:
        title = st.selectbox(
            "🔎 Search / choose a title",
            options=sorted(df["title"].tolist()),
            index=sorted(df["title"].tolist()).index("Midnight Mass"),
        )
    with col_n:
        top_n = st.slider("Number of recommendations", 5, 15, 10)

    picked = df[df["title"] == title].iloc[0]
    st.markdown(
        f"""
        <div class="picked">
          <b style="color:#e50914;">YOU PICKED:</b>
          <b>{html.escape(str(picked['title']))}</b> —
          {html.escape(str(picked['type']))} · {int(picked['release_year'])} ·
          {html.escape(str(picked['rating']))} · {html.escape(str(picked['listed_in']))}<br/>
          <span style="color:#a8a8a8;">{html.escape(describe(picked))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🎬 Get Recommendations", type="primary",
                 width="stretch"):
        recs = recommender.recommend(title, top_n=top_n)
        st.subheader(f"Because you watched “{title}”…")
        for start in range(0, len(recs), 2):
            cols = st.columns(2)
            for offset, col in enumerate(cols):
                idx = start + offset
                if idx < len(recs):
                    with col:
                        render_card(idx + 1, recs.iloc[idx])
        with st.expander("📋 View as a table"):
            table = recs.rename(columns={
                "title": "Title", "similarity": "Match score", "type": "Type",
                "director": "Director", "country": "Country",
                "release_year": "Year", "rating": "Rating",
                "duration": "Duration", "listed_in": "Genres"})
            st.dataframe(table, width="stretch", hide_index=True)

    with st.expander("ℹ️ How does it work?"):
        st.markdown(
            "Every title is converted into a text **profile** from its genres, "
            "country, director, type and rating. **TF-IDF** turns those profiles "
            "into vectors and **cosine similarity** ranks the catalogue by how "
            "close each title is to the one you picked — no user data needed, "
            "pure content-based filtering. Built as an end-to-end ML project."
        )

    st.markdown(
        """
        <div class="footer">
          Built by <b>Muhammad Abdul Rafay</b> — Machine Learning Intern |
          GitHub: <a href="https://github.com/mabdulrafay7zip">github.com/mabdulrafay7zip</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
