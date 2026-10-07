# 🎬 Netflix Recommendation System

A content-based movie & TV show recommender over the **8,790-title Netflix
catalogue**, built with **TF-IDF + cosine similarity** and served as an
interactive **Streamlit** web app. Pick any title and instantly get the most
similar titles, each with a match score, type, year, rating, genres and
director/country details.

**Live Demo:** (deployed on Streamlit Cloud)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B)
![scikit--learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange)

## How it works

1. Every title is turned into one short **text profile** built from its
   genres (`listed_in`), country, director, type and rating (multi-word
   names are squashed into single tokens, e.g. `MikeFlanagan`).
2. **TF-IDF** (`TfidfVectorizer`, English stop-words removed) vectorises all
   8,787 unique-title profiles — a vocabulary of ~5,400 terms.
3. **Cosine similarity** between the selected title's vector and the whole
   catalogue ranks the recommendations; the model is built once and cached.

No user ratings or history are needed — pure content-based filtering.

## Example

Asking for titles similar to **Midnight Mass** returns its Mike Flanagan /
horror neighbours at the top:

| # | Title | Match |
|---|-------|-------|
| 1 | The Haunting of Hill House | 72.7% |
| 2 | Ratched | 72.7% |
| 3 | The Haunting of Bly Manor | 72.7% |
| 4 | Brand New Cherry Flavor | 72.7% |
| 5 | The Originals | 68.8% |

## Project context

This app is the live-demo version of **Task 1** of the **Auspify
Technologies Machine Learning internship** (Oct–Nov 2026), where the same
recommender was first built as a standalone Python script
(`recommendation_system.py`). The full internship project set — content-type
prediction (91.9% accuracy), rating classification and K-Means content
segmentation — lives on GitHub:
[github.com/mabdulrafay7zip](https://github.com/mabdulrafay7zip)

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (default http://localhost:8501).

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit app + the TF-IDF recommender (cached data/model) |
| `data.csv` | Trimmed Netflix catalogue (8,790 rows: title, type, director, country, release year, rating, duration, genres) |
| `requirements.txt` | streamlit, pandas, scikit-learn |

## Author

**Muhammad Abdul Rafay** — BS Artificial Intelligence, Air University
Islamabad · ML Intern @ Auspify Technologies ·
[github.com/mabdulrafay7zip](https://github.com/mabdulrafay7zip)
