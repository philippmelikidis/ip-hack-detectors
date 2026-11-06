"""Streamlit labeling app (codebook v1.0).

    uv run --extra label streamlit run src/ip_hack_detectors/label_app.py -- \\
        --blinded grid_blinded.csv --labels-dir labels

The paths can also come from the environment (IPD_LABEL_BLINDED, IPD_LABEL_DIR). The app shows one
item at a time in the order of the blinded CSV, skips items the rater already labeled, and
appends every label at once to labels/labels_<rater>.csv.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import streamlit as st

from ip_hack_detectors.labeling import (
    CODEBOOK_PATH,
    LABELS,
    append_label,
    codebook_version,
    labeled_item_ids,
    read_blinded,
)

CODEBOOK_URL = "https://github.com/philippmelikidis/ip-hack-detectors/blob/main/docs/codebook.md"


def _paths() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--blinded", default=os.environ.get("IPD_LABEL_BLINDED"))
    parser.add_argument("--labels-dir", default=os.environ.get("IPD_LABEL_DIR", "labels"))
    args, _ = parser.parse_known_args()
    if not args.blinded:
        st.error("Start the app with `-- --blinded <blinded csv> [--labels-dir <dir>]`.")
        st.stop()
    return Path(args.blinded), Path(args.labels_dir)


def main() -> None:
    st.set_page_config(page_title="Label outputs", layout="wide")
    blinded_path, labels_dir = _paths()
    version = codebook_version()
    items = read_blinded(blinded_path)

    st.title("Label outputs")
    st.markdown(f"Codebook v{version}: [docs/codebook.md]({CODEBOOK_URL})")
    with st.expander("Show the codebook"):
        st.markdown(CODEBOOK_PATH.read_text(encoding="utf-8"))

    if "rater" not in st.session_state:
        st.text_input("Rater id", key="rater_id")
        if st.button("Start", key="start") and st.session_state["rater_id"].strip():
            st.session_state["rater"] = st.session_state["rater_id"].strip()
            st.rerun()
        st.stop()
    rater = st.session_state["rater"]

    done = labeled_item_ids(labels_dir, rater)
    todo = [i for i in items if i["item_id"] not in done]
    st.caption(f"Rater {rater}: {len(items) - len(todo)} of {len(items)} labeled")
    if not todo:
        st.success("All items are labeled. Thank you.")
        st.stop()
    item = todo[0]
    item_id = item["item_id"]
    st.session_state["current_item_id"] = item_id

    st.subheader("Task")
    st.text(item["task_text"])  # verbatim: Markdown would hide *, _, $ (codebook R6 needs them)
    left, right = st.columns(2)
    with left:
        st.subheader("Test 1 (shown to the model)")
        st.code(item["test_1"], language="python")
        st.subheader("All tests")
        st.code(item["all_tests"], language="python")
    with right:
        st.subheader("Extracted code")
        st.code(item["code"], language="python")

    free_text = st.text_area("Free text (required for unclear)", key=f"free_text_{item_id}")
    cols = st.columns(len(LABELS))
    for col, label in zip(cols, LABELS, strict=True):
        # keyed by item: a late or double click on an item already saved finds no button and is
        # dropped, instead of labeling the next item the rater has not seen
        if col.button(label, key=f"label_{item_id}_{label}", use_container_width=True):
            if label == "unclear" and not free_text.strip():
                st.error("Unclear needs a reason in the free text.")
                continue
            append_label(labels_dir, rater, version, item_id, label, free_text)
            st.rerun()


main()
