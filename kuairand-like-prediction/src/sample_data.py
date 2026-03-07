"""Generate a tiny synthetic dataset for fast local testing.

Creates CSVs: interactions.csv, users.csv, videos.csv, video_stats.csv
in the provided directory.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def generate_synthetic_sample(out_dir: Path, n_users: int = 50, n_videos: int = 200, n_interactions: int = 500):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    # users
    users = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n_users)],
        "user_age": rng.randint(18, 60, size=n_users),
        "user_country": rng.choice(["US", "IN", "BR", "ID"], size=n_users),
    })

    # videos
    videos = pd.DataFrame({
        "video_id": [f"v{i}" for i in range(n_videos)],
        "video_length": rng.randint(5, 300, size=n_videos),
        "video_category": rng.choice(["comedy", "music", "sports", "news"], size=n_videos),
    })

    # video_stats
    video_stats = pd.DataFrame({
        "video_id": videos["video_id"].values,
        "views": (rng.rand(n_videos) * 10000).astype(int),
        "likes": (rng.rand(n_videos) * 1000).astype(int),
    })

    # interactions
    user_choices = rng.choice(users["user_id"].values, size=n_interactions)
    video_choices = rng.choice(videos["video_id"].values, size=n_interactions)
    timestamps = pd.date_range("2020-01-01", periods=n_interactions, freq="T")
    # simple signal: shorter videos more likely liked by young users
    like_prob = []
    for u, v in zip(user_choices, video_choices):
        age = int(users.loc[users["user_id"] == u, "user_age"])
        length = int(videos.loc[videos["video_id"] == v, "video_length"])
        p = 0.2 + (40 - abs(age - 30)) / 200 + (1.0 - (length / 300.0)) * 0.2
        like_prob.append(min(max(p, 0.01), 0.99))
    likes = (rng.rand(n_interactions) < np.array(like_prob)).astype(int)

    interactions = pd.DataFrame({
        "user_id": user_choices,
        "video_id": video_choices,
        "timestamp": timestamps.astype(str),
        "is_like": likes,
    })

    users.to_csv(out_dir / "users.csv", index=False)
    videos.to_csv(out_dir / "videos.csv", index=False)
    video_stats.to_csv(out_dir / "video_stats.csv", index=False)
    interactions.to_csv(out_dir / "interactions.csv", index=False)

    return {
        "users": users,
        "videos": videos,
        "video_stats": video_stats,
        "interactions": interactions,
    }
