from typing import Dict, List
import re


class FeatureRegistry:
    """Simple registry to manage allowed and banned feature groups for leakage control.

    This class uses substring matching on column names to map them to groups. It's
    intentionally conservative: banned groups remove columns if any banned keyword
    matches the column name.
    """

    def __init__(self, allowed_groups: List[str] = None, banned_groups: List[str] = None):
        self.allowed_groups = allowed_groups or []
        self.banned_groups = banned_groups or []

    def is_banned(self, col: str) -> bool:
        col_l = col.lower()
        for k in self.banned_groups:
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
        """Partition columns into allowed and banned lists.

        Returns a dict with keys: 'allowed', 'banned', 'unknown'.
        """
        from typing import Dict, List
        import re
        import pandas as pd


        # Target and standard column groups used in the project
        TARGET_COLUMN = "is_like"
        # Primary identifier columns expected to join tables
        ID_COLUMNS = ["user_id", "video_id", "session_id"]
        # Known categorical columns (project-specific defaults). If missing in a dataset,
        # they are ignored — discovery falls back to dtype-based inference.
        CATEGORICAL_COLUMNS = [
            "user_country",
            "user_device",
            "video_category",
            "video_tags",
        ]
        # Known numeric columns commonly present in KuaiRand-like datasets.
        NUMERIC_COLUMNS = [
            "user_age",
            "video_duration",
            "video_like_count",
            "video_comment_count",
        ]
        # Contextual columns that are safe to use (pre-exposure context).
        CONTEXT_COLUMNS = [
            "session_id",
            "position_in_feed",
            "recommendation_algorithm",
        ]

        # BANNED_LEAKAGE_COLUMNS contains keywords or exact column names that indicate
        # post-exposure signals or alternative targets. These must not be used as
        # predictors for `is_like` because they leak information about the outcome.
        # Examples and rationale:
        # - 'click' / 'is_click' / 'clicked' : whether the user clicked the candidate — this
        #    happens after exposure and is a downstream outcome.
        # - 'watch_time' / 'watch_seconds' / 'watch_fraction' : watch metrics are post-exposure
        #    behavioral signals correlated with liking; including them would leak the label.
        # - 'exposure' / 'impression' : fields that encode the impression event (timestamp or id)
        #    may carry leakage if they represent the current exposure; use only exposure timestamp
        #    for time-splitting, not as a predictive feature.
        # - 'future_*' : any feature explicitly labeled as future, which by definition uses events
        #    after the exposure and so leaks.
        # - 'label_' : generic prefix for other labels that might directly encode targets.
        BANNED_LEAKAGE_COLUMNS = [
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
        ]


        class FeatureRegistry:
            """Registry to manage allowed and banned features for leakage control.

            The registry is intentionally simple: it performs substring matching of banned
            keywords against column names. This keeps policy explicit and auditable.
            """

            def __init__(self, allowed_groups: List[str] = None, banned_keywords: List[str] = None):
                self.allowed_groups = allowed_groups or []
                # Merge default banned keywords with any additional ones passed in
                self.banned_keywords = (banned_keywords or []) + BANNED_LEAKAGE_COLUMNS

            def is_banned(self, col: str) -> bool:
                """Return True if column name matches any banned keyword (case-insensitive)."""
                col_l = col.lower()
                for k in self.banned_keywords:
                    if k.lower() in col_l:
                        return True
                return False

            def filter_columns(self, cols: List[str]) -> Dict[str, List[str]]:
                """Partition columns into allowed and banned lists.

                Returns a dict with keys: 'allowed', 'banned', 'unknown'.
                """
                allowed = []
                banned = []
                unknown = []
                for c in cols:
                    if self.is_banned(c):
                        banned.append(c)
                    else:
                        # If allowed_groups configured, prefer that matching; otherwise treat as allowed
                        if not self.allowed_groups:
                            allowed.append(c)
                        else:
                            col_l = c.lower()
                            matched = any(g.lower() in col_l for g in self.allowed_groups)
                            if matched:
                                allowed.append(c)
                            else:
                                unknown.append(c)
                return {"allowed": allowed, "banned": banned, "unknown": unknown}


        def infer_feature_groups_from_patterns(patterns: Dict[str, str], cols: List[str]) -> Dict[str, List[str]]:
            """Assign columns to groups by regex patterns.

            patterns: mapping group -> regex
            Returns mapping group -> matched cols
            """
            out = {g: [] for g in patterns}
            for c in cols:
                for g, p in patterns.items():
                    if re.search(p, c):
                        out[g].append(c)
            return out


        def get_training_columns(df: pd.DataFrame, registry: FeatureRegistry = None) -> List[str]:
            """Return the list of columns safe to use for training.

            - Removes id columns and the target column.
            - Removes columns matching banned leakage keywords.
            - Preserves numeric and categorical columns; handles missing columns gracefully.

            Args:
                df: input DataFrame
                registry: optional FeatureRegistry; if None, a default one is used.

            Returns:
                list of column names suitable for training.
            """
            registry = registry or FeatureRegistry()
            cols = list(df.columns)
            # Exclude ids and target
            excluded = [c for c in ID_COLUMNS if c in cols] + ([TARGET_COLUMN] if TARGET_COLUMN in cols else [])
            candidate_cols = [c for c in cols if c not in excluded]
            parts = registry.filter_columns(candidate_cols)
            # Allowed + unknown (unknowns may be safe; caller can inspect)
            training_cols = parts.get("allowed", []) + parts.get("unknown", [])
            return training_cols


        def validate_no_banned_columns(df: pd.DataFrame, registry: FeatureRegistry = None) -> None:
            """Raise ValueError if any banned/leaky columns are present in df.

            The error message lists offending columns for easy inspection.
            """
            registry = registry or FeatureRegistry()
            cols = list(df.columns)
            parts = registry.filter_columns(cols)
            banned = parts.get("banned", [])
            if banned:
                raise ValueError(f"Banned leakage columns present in input: {banned}")
