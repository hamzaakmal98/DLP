from typing import Dict, List


# Standard names and defaults used across the repo
TARGET_COLUMN = "is_like"
ID_COLUMNS = ["user_id", "video_id", "session_id"]

# Default banned keywords indicating post-exposure leakage
DEFAULT_BANNED_KEYWORDS = [
    "click",
    "is_click",
    "clicked",
    "watch_time",
    "watch_seconds",
    "watch_percent",
    "watch_fraction",
    "exposure",
    "impression",
    "future",
    "label_",
    "outcome",
    "is_share",
    "is_subscribe",
    # interaction-level play/watch signals (likely post-exposure)
    "play_time",
    "play_time_ms",
    "duration_ms",
    "play_duration",
    "profile_stay_time",
    "comment_stay_time",
    "long_view",
    # other per-interaction labels
    "is_comment",
    "is_forward",
    "is_follow",
    "is_hate",
    "is_profile_enter",
]


class FeatureRegistry:
    """Registry to manage allowed and banned features for leakage control.

    Uses simple substring matching for banned keywords. Callers can pass
    additional `allowed_groups` or `banned_keywords` to customize behavior.
    """

    def __init__(self, allowed_groups: List[str] = None, banned_keywords: List[str] = None):
        self.allowed_groups = allowed_groups or []
        self.banned_keywords = (banned_keywords or []) + DEFAULT_BANNED_KEYWORDS

    def is_banned(self, col: str) -> bool:
        col_l = col.lower()
        for k in self.banned_keywords:
            if k.lower() in col_l:
                return True
        return False

    def is_allowed(self, col: str) -> bool:
        if not self.allowed_groups:
            return True
        col_l = col.lower()
        for k in self.allowed_groups:
            if k.lower() in col_l:
                return True
        return False

    def filter_columns(self, cols: List[str]) -> Dict[str, List[str]]:
        allowed = []
        banned = []
        unknown = []
        for c in cols:
            if self.is_banned(c):
                banned.append(c)
            else:
                if not self.allowed_groups:
                    allowed.append(c)
                else:
                    if self.is_allowed(c):
                        allowed.append(c)
                    else:
                        unknown.append(c)
        return {"allowed": allowed, "banned": banned, "unknown": unknown}


def get_training_columns(df, registry: FeatureRegistry = None) -> List[str]:
    registry = registry or FeatureRegistry()
    cols = list(df.columns)
    # Exclude id columns, the target, and common timestamp columns from training features
    excluded = [c for c in ID_COLUMNS if c in cols] + ([TARGET_COLUMN] if TARGET_COLUMN in cols else [])
    timestamp_candidates = [c for c in cols if c.lower() in ("timestamp", "time", "time_ms")]
    excluded += timestamp_candidates
    candidate_cols = [c for c in cols if c not in excluded]
    parts = registry.filter_columns(candidate_cols)
    training_cols = parts.get("allowed", []) + parts.get("unknown", [])
    return training_cols


def validate_no_banned_columns(df, registry: FeatureRegistry = None) -> None:
    registry = registry or FeatureRegistry()
    parts = registry.filter_columns(list(df.columns))
    banned = parts.get("banned", [])
    if banned:
        raise ValueError(f"Banned leakage columns present in input: {banned}")
