from __future__ import annotations

import html
import random
import re
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


DATA_FILE = Path(__file__).with_name("weekend_database.csv")
RESULTS_BATCH_SIZE = 20

BUDGET_CAPS = ["Any", "Free or $", "$$ max", "$$$ allowed"]
RAIN_FILTERS = ["Any", "Yes or maybe", "Yes only"]
SORT_OPTIONS = [
    "Best match",
    "Nearest first",
    "Kid score",
    "Adult score",
    "Lowest budget",
]

MODES = {
    "Balanced": {
        "distance_range": (0, 60),
        "budget_cap": "Any",
        "rain_filter": "Any",
        "pram_only": False,
    },
    "Quick local": {
        "distance_range": (0, 15),
        "budget_cap": "$$ max",
        "rain_filter": "Any",
        "pram_only": False,
    },
    "Low cost": {
        "distance_range": (0, 45),
        "budget_cap": "Free or $",
        "rain_filter": "Any",
        "pram_only": False,
    },
    "Pram friendly": {
        "distance_range": (0, 45),
        "budget_cap": "Any",
        "rain_filter": "Any",
        "pram_only": True,
    },
    "Rainy day": {
        "distance_range": (0, 45),
        "budget_cap": "Any",
        "rain_filter": "Yes only",
        "pram_only": False,
    },
    "Adult reset": {
        "distance_range": (0, 75),
        "budget_cap": "Any",
        "rain_filter": "Any",
        "pram_only": False,
    },
    "Big day trip": {
        "distance_range": (0, 120),
        "budget_cap": "Any",
        "rain_filter": "Any",
        "pram_only": False,
    },
}

TIME_GROUPS = {
    "Morning": ("morning", "early", "am"),
    "Afternoon": ("afternoon", "pm"),
    "Evening": ("evening", "night"),
    "Lunch": ("lunch",),
    "Rainy day": ("rain",),
}


def find_column(df: pd.DataFrame, name: str | None = None, contains: str | None = None) -> str:
    if name and name in df.columns:
        return name
    if contains:
        needle = contains.casefold()
        for column in df.columns:
            if needle in column.casefold():
                return column
    raise KeyError(name or contains or "column")


def budget_rank(value: str) -> float:
    text = str(value).strip()
    dollar_count = text.count("$")
    has_free = "free" in text.casefold()
    if has_free and dollar_count == 0:
        return 0.0
    if has_free and dollar_count <= 1:
        return 1.0
    if dollar_count:
        return float(dollar_count)
    return 2.0


def duration_rank(value: str) -> float:
    text = str(value).casefold()
    if "day" in text:
        return 8.0
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return 2.0
    return max(numbers)


@st.cache_data(show_spinner=False)
def load_data(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df = raw.copy()

    for column in df.columns:
        df[column] = df[column].astype(str).str.strip()

    columns = {
        "id": find_column(df, "ID"),
        "destination": find_column(df, "Destination"),
        "experience": find_column(df, contains="Experience"),
        "category": find_column(df, "Category"),
        "region": find_column(df, "Region"),
        "distance": find_column(df, "Approx km from Brisbane CBD"),
        "drive_time": find_column(df, "Approx drive time"),
        "distance_band": find_column(df, "Distance band"),
        "budget": find_column(df, "Budget"),
        "best_time": find_column(df, "Best time"),
        "duration": find_column(df, "Suggested duration"),
        "ages": find_column(df, "Suitable ages"),
        "pram": find_column(df, "Pram friendly"),
        "rainy": find_column(df, "Rainy-day friendly"),
        "adult_relax": find_column(df, "Adult relax value"),
        "kid_score": find_column(df, "Kid score /5"),
        "adult_score": find_column(df, "Adult score /5"),
        "priority": find_column(df, "Priority"),
        "parking": find_column(df, "Parking"),
        "toilets": find_column(df, "Toilets"),
        "route": find_column(df, "Suggested combo route"),
        "verify": find_column(df, "Verify before leaving"),
        "source": find_column(df, "Source URL"),
        "notes": find_column(df, "Notes"),
    }

    df.attrs["columns"] = columns
    df["_distance_km"] = pd.to_numeric(df[columns["distance"]], errors="coerce")
    df["_kid_score"] = pd.to_numeric(df[columns["kid_score"]], errors="coerce").fillna(0)
    df["_adult_score"] = pd.to_numeric(df[columns["adult_score"]], errors="coerce").fillna(0)
    df["_budget_rank"] = df[columns["budget"]].map(budget_rank).astype(float)
    df["_duration_rank"] = df[columns["duration"]].map(duration_rank).astype(float)

    search_columns = [
        columns["destination"],
        columns["experience"],
        columns["category"],
        columns["region"],
        columns["best_time"],
        columns["route"],
        columns["notes"],
    ]
    df["_search_text"] = (
        df[search_columns]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.casefold()
    )

    return df


def apply_mode_defaults() -> None:
    defaults = MODES[st.session_state.mode]
    for key, value in defaults.items():
        st.session_state[key] = value


def ensure_state() -> None:
    if "mode" not in st.session_state:
        st.session_state.mode = "Balanced"
    if "distance_limit" in st.session_state and "distance_range" not in st.session_state:
        st.session_state.distance_range = (0, st.session_state.distance_limit)
    for key, value in MODES[st.session_state.mode].items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("shortlist", [])
    st.session_state.setdefault("picked_id", None)
    st.session_state.setdefault("visible_destination_count", RESULTS_BATCH_SIZE)
    st.session_state.setdefault("result_signature", None)


def option_list(series: pd.Series) -> list[str]:
    return sorted(value for value in series.dropna().astype(str).unique() if value.strip())


def budget_allowed(df: pd.DataFrame, cap: str) -> pd.Series:
    if cap == "Any":
        return pd.Series(True, index=df.index)
    if cap == "Free or $":
        return df["_budget_rank"] <= 1
    if cap == "$$ max":
        return df["_budget_rank"] <= 2
    return df["_budget_rank"] <= 3


def time_allowed(series: pd.Series, selected_groups: list[str]) -> pd.Series:
    if not selected_groups:
        return pd.Series(True, index=series.index)

    def matches(value: str) -> bool:
        text = str(value).casefold()
        for group in selected_groups:
            if any(token.casefold() in text for token in TIME_GROUPS[group]):
                return True
        return False

    return series.map(matches)


def filter_data(
    df: pd.DataFrame,
    columns: dict[str, str],
    query: str,
    categories: list[str],
    regions: list[str],
    priorities: list[str],
    selected_times: list[str],
    age_filter: str,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    min_distance, max_distance = st.session_state.distance_range

    if query.strip():
        for term in query.casefold().split():
            mask &= df["_search_text"].str.contains(re.escape(term), na=False)

    mask &= df["_distance_km"].isna() | df["_distance_km"].between(min_distance, max_distance, inclusive="both")
    mask &= budget_allowed(df, st.session_state.budget_cap)

    if st.session_state.pram_only:
        mask &= df[columns["pram"]].eq("Yes")

    rain_filter = st.session_state.rain_filter
    if rain_filter == "Yes only":
        mask &= df[columns["rainy"]].eq("Yes")
    elif rain_filter == "Yes or maybe":
        mask &= df[columns["rainy"]].isin(["Yes", "Maybe"])

    if categories:
        mask &= df[columns["category"]].isin(categories)
    if regions:
        mask &= df[columns["region"]].isin(regions)
    if priorities:
        mask &= df[columns["priority"]].isin(priorities)

    mask &= time_allowed(df[columns["best_time"]], selected_times)

    if age_filter == "Baby/toddler included":
        mask &= df[columns["ages"]].str.contains("0-3", na=False)
    elif age_filter == "Preschool only":
        mask &= df[columns["ages"]].str.contains("3-6", na=False)
    elif age_filter == "Older kid friendly":
        mask &= df[columns["ages"]].str.contains("3-6\\+", regex=True, na=False)

    return df.loc[mask].copy()


def score_data(df: pd.DataFrame, columns: dict[str, str], mode: str) -> pd.DataFrame:
    if df.empty:
        df["_score"] = []
        df["_fit"] = []
        return df

    priority_bonus = df[columns["priority"]].map(
        {"A - easy repeat": 1.2, "B - planned day": 0.7, "C - bigger day trip": 0.3}
    ).fillna(0.5)
    relax_bonus = df[columns["adult_relax"]].eq("High").astype(float) * 0.8
    pram_bonus = df[columns["pram"]].map({"Yes": 0.5, "Maybe": 0.1, "No": -0.4}).fillna(0)
    rainy_bonus = df[columns["rainy"]].map({"Yes": 0.5, "Maybe": 0.1, "No": -0.3}).fillna(0)
    distance = df["_distance_km"].fillna(df["_distance_km"].median())

    score = (
        df["_kid_score"] * 1.5
        + df["_adult_score"] * 1.2
        + priority_bonus
        + relax_bonus
        + pram_bonus
        + rainy_bonus
        - df["_budget_rank"] * 0.15
        - distance.fillna(50) / 90
    )

    if mode == "Quick local":
        score += df[columns["priority"]].eq("A - easy repeat").astype(float) * 1.4
        score -= distance.fillna(50) / 22
        score -= df["_duration_rank"].clip(lower=0, upper=8) * 0.12
    elif mode == "Low cost":
        score -= df["_budget_rank"] * 0.9
        score += df[columns["budget"]].str.contains("Free", case=False, na=False).astype(float) * 1.2
    elif mode == "Pram friendly":
        score += df[columns["pram"]].map({"Yes": 2.0, "Maybe": 0.4, "No": -3.0}).fillna(0)
        score -= distance.fillna(50) / 70
    elif mode == "Rainy day":
        score += df[columns["rainy"]].map({"Yes": 3.0, "Maybe": 0.4, "No": -3.0}).fillna(0)
        score -= distance.fillna(50) / 70
    elif mode == "Adult reset":
        score += df[columns["adult_relax"]].eq("High").astype(float) * 2.0
        score += df["_adult_score"] * 0.7
    elif mode == "Big day trip":
        score += df[columns["priority"]].isin(["B - planned day", "C - bigger day trip"]).astype(float) * 1.1
        score += distance.fillna(0).between(30, 110).astype(float) * 1.0
        score -= distance.fillna(80) / 140

    ranked = df.copy()
    ranked["_score"] = score
    min_score = float(score.min())
    max_score = float(score.max())
    if max_score == min_score:
        ranked["_fit"] = 85
    else:
        ranked["_fit"] = (((score - min_score) / (max_score - min_score)) * 30 + 70).round()
    return ranked


def sort_data(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if df.empty:
        return df
    if sort_by == "Nearest first":
        return df.sort_values(["_distance_km", "_score"], ascending=[True, False], na_position="last")
    if sort_by == "Kid score":
        return df.sort_values(["_kid_score", "_score"], ascending=[False, False])
    if sort_by == "Adult score":
        return df.sort_values(["_adult_score", "_score"], ascending=[False, False])
    if sort_by == "Lowest budget":
        return df.sort_values(["_budget_rank", "_score"], ascending=[True, False])
    return df.sort_values(["_score", "_kid_score", "_adult_score"], ascending=[False, False, False])


def destination_names(df: pd.DataFrame, columns: dict[str, str], limit: int | None = None) -> list[str]:
    names = df[columns["destination"]].drop_duplicates().tolist()
    return names if limit is None else names[:limit]


def compact_values(series: pd.Series, limit: int = 3) -> str:
    values = [str(value).strip() for value in series.dropna().unique() if str(value).strip()]
    if not values:
        return "-"
    if len(values) <= limit:
        return " / ".join(values)
    return " / ".join(values[:limit]) + f" +{len(values) - limit}"


def maps_query(row: pd.Series, columns: dict[str, str]) -> str:
    parts = [
        row[columns["destination"]],
        row[columns["region"]],
        "Queensland Australia",
    ]
    return ", ".join(str(part).strip() for part in parts if str(part).strip())


def render_map(row: pd.Series, columns: dict[str, str]) -> None:
    query = maps_query(row, columns)
    src = f"https://www.google.com/maps?q={quote_plus(query)}&output=embed"
    escaped_query = html.escape(query)
    components.html(
        f"""
        <iframe
            title="Map for {escaped_query}"
            src="{src}"
            width="100%"
            height="280"
            style="border:0; border-radius:8px;"
            loading="lazy"
            referrerpolicy="no-referrer-when-downgrade">
        </iframe>
        """,
        height=295,
    )


def key_token(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")[:90] or "item"


def badge(label: object, tone: str = "neutral") -> str:
    return f'<span class="badge badge-{tone}">{html.escape(str(label))}</span>'


def add_to_shortlist(item_id: str) -> None:
    if item_id not in st.session_state.shortlist:
        st.session_state.shortlist.append(item_id)


def remove_from_shortlist(item_id: str) -> None:
    st.session_state.shortlist = [value for value in st.session_state.shortlist if value != item_id]
    if st.session_state.picked_id == item_id:
        st.session_state.picked_id = None


def render_activity_list(group: pd.DataFrame, columns: dict[str, str]) -> None:
    st.markdown("**Experience / Activity**")
    for _, row in group.iterrows():
        meta = " | ".join(
            value
            for value in [
                row[columns["best_time"]],
                row[columns["duration"]],
                row[columns["budget"]],
                f"Kid {row[columns['kid_score']]}/5",
                f"Adult {row[columns['adult_score']]}/5",
            ]
            if value
        )
        st.markdown(
            f"""
            <div class="activity-row">
                <div class="activity-title">{html.escape(row[columns["experience"]])}</div>
                <div class="activity-meta">{html.escape(meta)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_destination_group(group: pd.DataFrame, columns: dict[str, str], key_prefix: str) -> None:
    if "_score" in group.columns:
        group = group.sort_values(["_score", "_kid_score", "_adult_score"], ascending=[False, False, False])
    lead = group.iloc[0]
    item_id = lead[columns["id"]]
    token = key_token(f"{key_prefix}_{lead[columns['destination']]}")
    is_saved = item_id in st.session_state.shortlist

    with st.container(border=True):
        title_column, score_column = st.columns([5, 1])
        with title_column:
            st.markdown(f"### {html.escape(lead[columns['destination']])}")
            st.caption(f"{len(group)} activity option{'s' if len(group) != 1 else ''}")
            badges = [
                badge(compact_values(group[columns["category"]]), "blue"),
                badge(compact_values(group[columns["region"]]), "neutral"),
                badge(compact_values(group[columns["drive_time"]]), "green"),
                badge(compact_values(group[columns["budget"]]), "amber"),
                badge(compact_values(group[columns["priority"]]), "neutral"),
            ]
            st.markdown(" ".join(badges), unsafe_allow_html=True)
        with score_column:
            st.metric("Fit", f"{int(lead['_fit'])}%")

        detail_columns = st.columns(4)
        detail_columns[0].metric("Best kid", f"{group['_kid_score'].max():.1f}/5")
        detail_columns[1].metric("Best adult", f"{group['_adult_score'].max():.1f}/5")
        detail_columns[2].metric("Pram", compact_values(group[columns["pram"]], limit=2))
        detail_columns[3].metric("Rain", compact_values(group[columns["rainy"]], limit=2))

        body_columns = st.columns([1.25, 1])
        with body_columns[0]:
            render_activity_list(group, columns)

            routes = compact_values(group[columns["route"]], limit=4)
            if routes and routes != "-":
                st.write(routes)
        with body_columns[1]:
            st.markdown("**Map**")
            render_map(lead, columns)

        with st.expander("Details"):
            st.write(f"Best time: {compact_values(group[columns['best_time']], limit=5)}")
            st.write(f"Suggested duration: {compact_values(group[columns['duration']], limit=5)}")
            st.write(f"Suitable ages: {compact_values(group[columns['ages']], limit=5)}")
            st.write(f"Parking: {compact_values(group[columns['parking']], limit=2)}")
            st.write(f"Toilets: {compact_values(group[columns['toilets']], limit=2)}")
            st.write(f"Verify: {compact_values(group[columns['verify']], limit=3)}")
            notes = compact_values(group[columns["notes"]], limit=3)
            if notes != "-":
                st.write(f"Notes: {notes}")

        action_columns = st.columns([1, 1, 4])
        if is_saved:
            if action_columns[0].button("Remove", key=f"{token}_remove"):
                remove_from_shortlist(item_id)
                st.rerun()
        else:
            if action_columns[0].button("Shortlist best", key=f"{token}_add"):
                add_to_shortlist(item_id)
                st.rerun()

        source_url = lead[columns["source"]]
        if source_url:
            action_columns[1].link_button("Source", source_url)


def render_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
            width: 100%;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
            background: #fbfbf9;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }
        .badge {
            display: inline-block;
            border: 1px solid #d7dce2;
            border-radius: 999px;
            padding: 0.18rem 0.52rem;
            margin: 0.12rem 0.18rem 0.12rem 0;
            font-size: 0.78rem;
            line-height: 1.2;
            background: #f8fafc;
            color: #26313d;
        }
        .badge-blue {
            background: #eef6ff;
            border-color: #bfdbfe;
            color: #1d4e89;
        }
        .badge-green {
            background: #effaf3;
            border-color: #bbdfc8;
            color: #1f6b42;
        }
        .badge-amber {
            background: #fff7e8;
            border-color: #f5d29b;
            color: #805400;
        }
        .activity-row {
            border-top: 1px solid #edf0f2;
            padding: 0.5rem 0;
        }
        .activity-title {
            color: #17202a;
            font-weight: 650;
            line-height: 1.25;
        }
        .activity-meta {
            color: #5f6b77;
            font-size: 0.84rem;
            margin-top: 0.12rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Weekend Picker", layout="wide")
    render_css()
    ensure_state()

    if not DATA_FILE.exists():
        st.error(f"Missing CSV file: {DATA_FILE.name}")
        return

    df = load_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
    columns = df.attrs["columns"]

    st.title("Weekend Picker")
    st.caption(
        f"{len(df):,} ideas from {DATA_FILE.name}."
    )

    with st.sidebar:
        st.header("Options")
        st.radio(
            "Weekend style",
            list(MODES),
            key="mode",
            on_change=apply_mode_defaults,
        )
        max_distance = int(max(120, df["_distance_km"].max(skipna=True)))
        st.slider(
            "Distance from Brisbane CBD",
            min_value=0,
            max_value=max_distance,
            step=5,
            key="distance_range",
        )
        st.selectbox("Budget ceiling", BUDGET_CAPS, key="budget_cap")
        st.selectbox("Rain plan", RAIN_FILTERS, key="rain_filter")

        st.divider()
        st.subheader("Filters")
        query = st.text_input(
            "Search",
            placeholder="Try beach, museum, market, Tamborine...",
        )
        age_filter = st.selectbox(
            "Age fit",
            ["Any", "Baby/toddler included", "Preschool only", "Older kid friendly"],
        )
        sort_by = st.selectbox("Sort", SORT_OPTIONS)

        category_options = option_list(df[columns["category"]])
        region_options = option_list(df[columns["region"]])
        priority_options = option_list(df[columns["priority"]])

        categories = st.multiselect("Category", category_options)
        regions = st.multiselect("Region", region_options)
        priorities = st.multiselect("Priority", priority_options)
        selected_times = st.multiselect("Best time", list(TIME_GROUPS))

    filtered = filter_data(
        df,
        columns,
        query=query,
        categories=categories,
        regions=regions,
        priorities=priorities,
        selected_times=selected_times,
        age_filter=age_filter,
    )
    ranked = sort_data(score_data(filtered, columns, st.session_state.mode), sort_by)
    result_signature = (
        st.session_state.mode,
        tuple(st.session_state.distance_range),
        st.session_state.budget_cap,
        st.session_state.rain_filter,
        query,
        age_filter,
        sort_by,
        tuple(categories),
        tuple(regions),
        tuple(priorities),
        tuple(selected_times),
    )
    if st.session_state.result_signature != result_signature:
        st.session_state.visible_destination_count = RESULTS_BATCH_SIZE
        st.session_state.result_signature = result_signature

    unique_destinations = ranked[columns["destination"]].nunique() if not ranked.empty else 0
    summary_columns = st.columns(4)
    summary_columns[0].metric("Matching ideas", f"{len(ranked):,}")
    summary_columns[1].metric("Destinations", f"{unique_destinations:,}")
    summary_columns[2].metric("Shortlist", len(st.session_state.shortlist))
    nearest = ranked["_distance_km"].min(skipna=True) if not ranked.empty else None
    summary_columns[3].metric("Nearest", f"{nearest:.0f} km" if nearest is not None else "-")

    tab_results, tab_shortlist, tab_table = st.tabs(["Recommended", "Shortlist", "Table"])

    with tab_results:
        top_actions = st.columns([1, 1, 4])
        representatives = ranked.drop_duplicates(subset=[columns["destination"]], keep="first")
        if top_actions[0].button("Random pick", disabled=representatives.empty):
            pool = representatives
            st.session_state.picked_id = random.choice(pool[columns["id"]].tolist())
            add_to_shortlist(st.session_state.picked_id)
            st.rerun()
        if top_actions[1].button("Clear shortlist", disabled=not st.session_state.shortlist):
            st.session_state.shortlist = []
            st.session_state.picked_id = None
            st.rerun()

        if st.session_state.picked_id:
            picked = df.loc[df[columns["id"]].eq(st.session_state.picked_id)]
            if not picked.empty:
                st.subheader("Picked option")
                picked_destination = picked.iloc[0][columns["destination"]]
                picked_group = ranked.loc[ranked[columns["destination"]].eq(picked_destination)].copy()
                if picked_group.empty:
                    picked_group = score_data(
                        df.loc[df[columns["destination"]].eq(picked_destination)].copy(),
                        columns,
                        st.session_state.mode,
                    )
                render_destination_group(picked_group, columns, "picked")

        if ranked.empty:
            st.info("No matches. Loosen the distance, budget, rain, or advanced filters.")
        else:
            st.subheader("Best matches")
            all_destinations = destination_names(ranked, columns)
            visible_count = min(st.session_state.visible_destination_count, len(all_destinations))
            visible_destinations = all_destinations[:visible_count]
            st.caption(f"Showing {visible_count} of {len(all_destinations)} destinations.")
            for destination in visible_destinations:
                group = ranked.loc[ranked[columns["destination"]].eq(destination)]
                render_destination_group(group, columns, "result")
            if visible_count < len(all_destinations):
                remaining = len(all_destinations) - visible_count
                next_count = min(RESULTS_BATCH_SIZE, remaining)
                if st.button(f"Load {next_count} more", key="load_more_results"):
                    st.session_state.visible_destination_count = visible_count + RESULTS_BATCH_SIZE
                    st.rerun()

    with tab_shortlist:
        saved = ranked.loc[ranked[columns["id"]].isin(st.session_state.shortlist)].copy()
        missing_saved = df.loc[
            df[columns["id"]].isin(st.session_state.shortlist)
            & ~df[columns["id"]].isin(saved[columns["id"]])
        ].copy()
        if not missing_saved.empty:
            missing_saved = score_data(missing_saved, columns, st.session_state.mode)
            saved = pd.concat([saved, missing_saved], ignore_index=True)

        if saved.empty:
            st.info("No shortlisted ideas yet.")
        else:
            compare_columns = [
                columns["destination"],
                columns["experience"],
                columns["region"],
                columns["drive_time"],
                columns["budget"],
                columns["pram"],
                columns["rainy"],
                columns["kid_score"],
                columns["adult_score"],
                columns["source"],
            ]
            st.dataframe(
                saved[compare_columns],
                use_container_width=True,
                hide_index=True,
                column_config={columns["source"]: st.column_config.LinkColumn("Source")},
            )
            if st.button("Pick from shortlist"):
                st.session_state.picked_id = random.choice(saved[columns["id"]].tolist())
                st.rerun()
            for destination in destination_names(saved, columns):
                group = saved.loc[saved[columns["destination"]].eq(destination)]
                render_destination_group(group, columns, "shortlist")

    with tab_table:
        table_columns = [
            columns["id"],
            columns["destination"],
            columns["experience"],
            columns["category"],
            columns["region"],
            columns["distance"],
            columns["drive_time"],
            columns["budget"],
            columns["best_time"],
            columns["duration"],
            columns["pram"],
            columns["rainy"],
            columns["kid_score"],
            columns["adult_score"],
            columns["priority"],
            columns["source"],
        ]
        table = ranked[table_columns].copy()
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={columns["source"]: st.column_config.LinkColumn("Source")},
        )
        st.download_button(
            "Download current matches",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="weekend_matches.csv",
            mime="text/csv",
            disabled=table.empty,
        )


if __name__ == "__main__":
    main()
