"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

from collections import Counter, defaultdict
from datetime import timedelta
from statistics import median
from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest

mcp = FastMCP("sf-311")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}

CountGroup = Literal["neighborhood", "service", "agency", "source", "weekday", "status"]
MedianGroup = Literal["neighborhood", "agency"]
Service = Annotated[str | None, Field(description="Exact service name from dataset_info, e.g. 'Graffiti Public'.")]
Neighborhood = Annotated[
    str | None, Field(description="Exact official neighborhood name from dataset_info, e.g. 'Tenderloin'.")
]


def _check(value: str | None, choices: list[str], kind: str) -> None:
    if value is None or value in choices:
        return
    suggestions = closest(value, choices)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ToolError(f"Unknown {kind} {value!r}.{hint} Call dataset_info for the full list of {kind} names.")


def _matching(service: str | None = None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    _check(service, SERVICES, "service")
    _check(neighborhood, NEIGHBORHOODS, "neighborhood")
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def search_requests(
    service: Service = None,
    neighborhood: Neighborhood = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: Annotated[int, Field(ge=1, le=MAX_ROWS)] = 20,
) -> list[dict]:
    """List individual 311 requests matching the filters, newest first.

    Returns up to `limit` rows (max 50), each with id, opened/closed times,
    status, service, subtype, agency, neighborhood, local name, address and
    source. Use this to look at examples. Do NOT use it to count or rank:
    it only returns a sample, so use count_requests for "how many" or
    "which has the most", and median_close_hours for "how fast".
    """
    rows = sorted(_matching(service, neighborhood, status), key=lambda r: r.opened, reverse=True)
    return [r.as_dict() for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: Service = None,
    neighborhood: Neighborhood = None,
    group_by: CountGroup | None = None,
) -> int | dict[str, int]:
    """Count requests over the whole week, optionally broken down by a field.

    Without group_by, returns one number. With group_by, returns a dict of
    group -> count, largest first, at most 50 groups; requests with an empty
    value for that field are left out. Use this for "how many" and "which X
    had the most" questions (e.g. service='Graffiti Public',
    group_by='neighborhood'). For response times use median_close_hours.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(service: Service = None, group_by: MedianGroup | None = None) -> float | None | dict[str, float]:
    """Median hours from opened to closed, for closed requests only, rounded to 0.1.

    Without group_by, returns one number (None if nothing has closed). With
    group_by ('neighborhood' or 'agency'), returns a dict of group -> median
    hours, fastest first, skipping groups with fewer than 10 closed requests.
    Use this for "how fast" or "who is slowest" questions; use
    count_requests for volumes.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {g: round(median(v), 1) for g, v in hours.items() if len(v) >= MIN_CLOSED_FOR_MEDIAN}
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(request_id: str) -> dict:
    """Return every field of one request by its request_id.

    Use this when you already have an id (from search_requests or the user).
    To find requests, use search_requests instead.
    """
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(f"No request with id {request_id!r}. Use search_requests to find valid ids.")
    return request.as_dict()


CHART_HEIGHT = 12


def _bar_chart(counts: list[tuple[str, int]], title: str) -> str:
    """Draw a vertical ASCII bar chart: x axis = labels, y axis = counts."""
    top = max((n for _, n in counts), default=0) or 1
    width = max(len(label) for label, _ in counts)
    y_width = len(str(top))
    lines = [title, ""]
    for level in range(CHART_HEIGHT, 0, -1):
        threshold = top * level / CHART_HEIGHT
        tick = str(round(threshold)) if level in (CHART_HEIGHT, CHART_HEIGHT // 2) else ""
        bars = " ".join(("█" * width if n >= threshold - top / (2 * CHART_HEIGHT) else " " * width) for _, n in counts)
        lines.append(f"{tick:>{y_width}} │ {bars}")
    lines.append(f"{'0':>{y_width}} └─" + "─" * (len(counts) * (width + 1)))
    lines.append(" " * (y_width + 3) + " ".join(label.center(width) for label, _ in counts))
    lines.append(" " * (y_width + 3) + " ".join(str(n).center(width) for _, n in counts))
    return "\n".join(lines)


@mcp.tool(annotations={"readOnlyHint": True})
def request_timeseries(neighborhood: Neighborhood = None) -> str:
    """Daily time series of 311 requests, drawn as a text bar chart.

    Returns a ready-to-print ASCII chart with one bar per day (x axis =
    date, y axis = number of requests opened that day), with the exact
    counts under each bar. Leave neighborhood empty for San Francisco as a
    whole, or pass an exact official neighborhood name for one area.
    Show the chart to the user verbatim in a code block. The data covers
    one week, so there is one bar per day, not per month. For a single
    total or a ranking across neighborhoods, use count_requests instead.
    """
    rows = _matching(neighborhood=neighborhood)
    if not rows:
        raise ToolError(f"No requests in {neighborhood!r} this week. Try another neighborhood or leave it empty for all of SF.")
    first = min(r.opened for r in REQUESTS).date()
    last = max(r.opened for r in REQUESTS).date()
    per_day = Counter(r.opened.date() for r in rows)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    counts = [(day.strftime("%a %d"), per_day.get(day, 0)) for day in days]
    area = neighborhood or "San Francisco (citywide)"
    title = f"311 requests per day — {area}, {first:%b %d}–{last:%b %d, %Y} (total {len(rows):,})"
    return _bar_chart(counts, title)


if __name__ == "__main__":
    mcp.run()
