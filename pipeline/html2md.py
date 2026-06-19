from html.parser import HTMLParser

def html_table_to_markdown(html):
    """Convert HTML table to Markdown table format."""
    state = {"rows": [], "current_row": [], "cell": "", "in_td": False, "tag_stack": []}

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag.lower() == "tr":
                state["current_row"] = []
            elif tag.lower() in ("td", "th"):
                state["in_td"] = True
                state["cell"] = ""
                state["tag_stack"] = [tag.lower()]
            elif state["in_td"]:
                state["tag_stack"].append(tag.lower())

        def handle_endtag(self, tag):
            if tag.lower() in ("td", "th"):
                state["current_row"].append(state["cell"].strip())
                state["in_td"] = False
                state["tag_stack"] = []
            elif tag.lower() == "tr":
                if state["current_row"]:
                    state["rows"].append(state["current_row"])
                state["current_row"] = []
            elif state["in_td"] and state["tag_stack"]:
                if state["tag_stack"][-1] == tag.lower():
                    state["tag_stack"].pop()

        def handle_data(self, data):
            if state["in_td"]:
                state["cell"] += data

    P().feed(html)

    if state["current_row"]:
        state["rows"].append(state["current_row"])

    rows = state["rows"]
    if not rows:
        return html

    max_cols = max(len(r) for r in rows) if rows else 0
    if max_cols == 0:
        return html

    for r in rows:
        while len(r) < max_cols:
            r.append("")

    lines = []
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
    for r in rows[1:]:
        lines.append("| " + " | ".join(r) + " |")

    return "\n".join(lines)
