"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from datetime import timedelta
from statistics import median
from typing import Annotated, Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ElicitRequest, ElicitRequestFormParams, InputRequiredResult
from pydantic import Field

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import ask_choice, caller_email, read_choice

mcp = FastMCP("sf-311")
log = logging.getLogger("sf311.server")

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


HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")


def mask_address(address: str) -> str:
    """Cut an address to its block: '994 HARRISON ST, SAN FRANCISCO, ...' -> '900 block of HARRISON ST'.

    Addresses without a leading house number (e.g. intersections) are returned unchanged.
    """
    match = HOUSE_NUMBER.match(address)
    if not match:
        return address
    return f"{int(match.group(1)) // 100 * 100} block of {match.group(2).strip()}"


def _public(request: Request) -> dict:
    """A request as a dict, with the address masked to the block. Use this for every row a tool returns."""
    row = request.as_dict()
    row["address"] = mask_address(row["address"])
    return row


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
    return [_public(r) for r in rows[:limit]]


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
    log.info("get_request request_id=%s caller=%s", request_id.strip(), caller_email() or "local")
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(f"No request with id {request_id!r}. Use search_requests to find valid ids.")
    return _public(request)


@mcp.tool(annotations={"readOnlyHint": True})
def open_backlog(
    service: Service = None,
    neighborhood: Neighborhood = None,
    group_by: Literal["neighborhood", "service", "agency", "source"] | None = None,
) -> dict:
    """How many requests opened this week are still open, i.e. the unresolved backlog.

    Use this for "how many are still open / unresolved / not yet closed"
    questions. Without group_by, returns {opened, still_open, closed,
    open_share, oldest_open_since} for the filters. With group_by, returns
    {group: still_open}, largest first, at most 50 groups. For totals of all
    requests use count_requests; for close times use median_close_hours.
    """
    rows = _matching(service, neighborhood)
    still_open = [r for r in rows if r.closed is None]
    if group_by is not None:
        key = GROUPS[group_by]
        counts = Counter(key(r) for r in still_open if key(r))
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_ROWS])
    return {
        "opened": len(rows),
        "still_open": len(still_open),
        "closed": len(rows) - len(still_open),
        "open_share": round(len(still_open) / len(rows), 3) if rows else None,
        "oldest_open_since": min(r.opened for r in still_open).isoformat() if still_open else None,
    }


FILTER_KEY = "filter_by"
FILTER_OPTIONS = {"Service": ("service",), "Neighborhood": ("neighborhood",), "Both": ("service", "neighborhood")}
PICK_KEYS = {"pick_service": ("service",), "pick_neighborhood": ("neighborhood",), "pick_both": ("service", "neighborhood")}
FIELD_CHOICES = {"service": SERVICES, "neighborhood": NEIGHBORHOODS}

def _ask_fields(key: str, message: str, fields: list[str]) -> InputRequiredResult:
    """Ask the user to pick a value for each field in one form (like ask_choice, but several fields)."""
    form = ElicitRequestFormParams(
        message=message,
        requested_schema={
            "type": "object",
            "properties": {f: {"type": "string", "title": f.title(), "enum": FIELD_CHOICES[f]} for f in fields},
            "required": fields,
        },
    )
    return InputRequiredResult(
        result_type="input_required",
        input_requests={key: ElicitRequest(method="elicitation/create", params=form)},
    )


CHART_ROWS = 12
DAY_SPACING = 8
POINT = "●"
BRAILLE_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
"""Braille dot bit for (subpixel row 0-3, subpixel column 0-1) inside one character cell."""


def _line_chart(counts: list[tuple[str, int]], title: str) -> str:
    """Draw a text line chart: a dotted braille line through ● points, box-drawing axes."""
    values = [n for _, n in counts]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    width = (len(values) - 1) * DAY_SPACING + 1
    px_h = CHART_ROWS * 4 - 1
    dots = [[0] * width for _ in range(CHART_ROWS)]

    pts = [(i * DAY_SPACING * 2, round((hi - v) / span * px_h)) for i, v in enumerate(values)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        steps = max(abs(x1 - x0), abs(y1 - y0))
        for k in range(steps + 1):
            x, y = round(x0 + (x1 - x0) * k / steps), round(y0 + (y1 - y0) * k / steps)
            dots[y // 4][min(x // 2, width - 1)] |= BRAILLE_BITS[y % 4][x % 2]

    grid = [[chr(0x2800 + bits) if bits else " " for bits in row] for row in dots]
    for x, y in pts:
        grid[y // 4][min(x // 2, width - 1)] = POINT

    mid = (CHART_ROWS - 1) // 2
    ticks = {0: hi, mid: round(hi - span * (mid * 4 + 1.5) / px_h), CHART_ROWS - 1: lo}
    y_width = len(f"{hi:,}")
    lines = [title, ""]
    for r, row in enumerate(grid):
        label = f"{ticks[r]:,}" if r in ticks else ""
        lines.append(f"{label:>{y_width}} {'┤' if r in ticks else '│'} " + "".join(row).rstrip())
    lines.append(" " * y_width + " └─" + "".join("┬" if c % DAY_SPACING == 0 else "─" for c in range(width)))
    pad = " " * (y_width + 3 - DAY_SPACING // 2 + 1)
    lines.append(pad + "".join(label.center(DAY_SPACING) for label, _ in counts).rstrip())
    lines.append(pad + "".join(f"{n:,}".center(DAY_SPACING) for _, n in counts).rstrip())
    lines.append("")
    lines.append(f"● = requests opened that day. The y axis runs from {lo:,} to {hi:,}, not from 0.")
    return "\n".join(lines)


@mcp.tool(annotations={"readOnlyHint": True})
def request_timeseries(
    ctx: Context, service: Service = None, neighborhood: Neighborhood = None
) -> str | InputRequiredResult:
    """Daily time series of 311 requests, drawn as a text line chart.

    Asks the user whether to filter by service, neighborhood, or both, then
    asks for whichever names you did not pass. Pass service and/or
    neighborhood (exact names from dataset_info) if the user already named
    them. Returns a ready-to-print chart with one point per day (x axis =
    date, y axis = requests opened that day) and the exact daily counts.
    Show it to the user verbatim inside a code block. The data covers one
    week, so there is one point per day, not per month. For totals or
    rankings use count_requests instead.
    """
    _check(service, SERVICES, "service")
    _check(neighborhood, NEIGHBORHOODS, "neighborhood")
    given = {"service": service, "neighborhood": neighborhood}
    responses = ctx.input_responses or {}

    picked = next((k for k in PICK_KEYS if k in responses), None)
    if picked:
        answer = responses[picked]
        if answer.action != "accept":
            return "No name chosen, so no chart was drawn."
        fields = PICK_KEYS[picked]
        chosen = {f: given[f] or answer.content.get(f) for f in fields}
    else:
        choice = read_choice(ctx, FILTER_KEY)
        if choice is None:
            return ask_choice(FILTER_KEY, "Filter the 311 time series by:", list(FILTER_OPTIONS))
        if not choice.accepted:
            return "No filter chosen, so no chart was drawn."
        fields = FILTER_OPTIONS[choice.value]
        chosen = {f: given[f] for f in fields}
        missing = [f for f in fields if not chosen[f]]
        if missing:
            key = "pick_both" if len(fields) == 2 else f"pick_{fields[0]}"
            return _ask_fields(key, f"Pick the {' and '.join(missing)} to chart:", missing)

    rows = _matching(service=chosen.get("service"), neighborhood=chosen.get("neighborhood"))
    if not rows:
        raise ToolError("No requests match that service and neighborhood this week. Try a different pair.")
    first = min(r.opened for r in REQUESTS).date()
    last = max(r.opened for r in REQUESTS).date()
    per_day = Counter(r.opened.date() for r in rows)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    counts = [(day.strftime("%a %d"), per_day.get(day, 0)) for day in days]
    label = " in ".join(v for v in (chosen.get("service"), chosen.get("neighborhood")) if v)
    title = f"311 requests per day — {label}, {first:%b %d}–{last:%b %d, %Y} (total {len(rows):,})"
    return _line_chart(counts, title)


if __name__ == "__main__":
    mcp.run()
