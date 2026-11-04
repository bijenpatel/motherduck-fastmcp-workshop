"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

from collections import Counter, defaultdict
from statistics import median
from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, closest

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


def _validate_service(service: str | None) -> None:
    if service is not None and service not in SERVICES:
        suggestions = closest(service, SERVICES)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ToolError(f"Unknown service '{service}'.{hint}")


def _validate_neighborhood(neighborhood: str | None) -> None:
    if neighborhood is not None and neighborhood not in NEIGHBORHOODS:
        suggestions = closest(neighborhood, NEIGHBORHOODS)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ToolError(f"Unknown neighborhood '{neighborhood}'.{hint}")


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
    service: str | None = None,
    neighborhood: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Search individual 311 service requests matching filters, newest first.

    Returns up to 50 requests as dictionaries with all request fields.
    Use this when you need detailed records or specific examples.
    For counting or ranking by categories, use count_requests instead.
    For resolution durations, use median_close_hours.
    """
    _validate_service(service)
    _validate_neighborhood(neighborhood)
    if status is not None and status not in ("Open", "Closed"):
        raise ToolError(f"Unknown status '{status}'. Choose 'Open' or 'Closed'.")

    limit = min(max(1, limit), MAX_ROWS)
    rows = [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]
    rows.sort(key=lambda r: r.opened, reverse=True)
    return [r.as_dict() for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: Literal["neighborhood", "service", "agency", "source", "weekday", "status"] | None = None,
) -> int | dict[str, int]:
    """Count service requests matching filters, optionally grouped by a category.

    Returns a total count number, or a dictionary mapping group names to counts
    (sorted by highest count first, up to 50 groups).
    Use this to aggregate volumes, find the most common services, or compare neighborhoods.
    To inspect individual request records, use search_requests or get_request instead.
    """
    _validate_service(service)
    _validate_neighborhood(neighborhood)
    if group_by is not None and group_by not in GROUPS:
        raise ToolError(f"Unknown group_by '{group_by}'. Choose one of: {', '.join(GROUPS.keys())}.")

    rows = [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
    ]

    if group_by is None:
        return len(rows)

    extractor = GROUPS[group_by]
    counts = Counter(extractor(r) for r in rows if extractor(r))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(
    service: str | None = None,
    group_by: Literal["neighborhood", "agency"] | None = None,
) -> float | None | dict[str, float]:
    """Calculate the median hours to resolve closed service requests.

    Returns a single median number rounded to 0.1 hours, or a dictionary of
    group names to median hours (fastest resolution first, at most 50 groups).
    Groups with fewer than 10 closed requests are excluded.
    Use this to analyze service response and resolution speed.
    To count request volumes, use count_requests instead.
    """
    _validate_service(service)
    if group_by is not None and group_by not in ("neighborhood", "agency"):
        raise ToolError(f"Unknown group_by '{group_by}'. Choose 'neighborhood' or 'agency'.")

    rows = [r for r in REQUESTS if (service is None or r.service == service) and r.hours_to_close is not None]

    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None

    extractor = GROUPS[group_by]
    hours_by_group: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        val = extractor(r)
        if val:
            hours_by_group[val].append(r.hours_to_close)

    medians = {
        group: round(median(values), 1)
        for group, values in hours_by_group.items()
        if len(values) >= MIN_CLOSED_FOR_MEDIAN
    }
    ranked = sorted(medians.items(), key=lambda item: (item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(request_id: str) -> dict:
    """Retrieve full details for a single service request by its ID.

    Returns the complete dictionary representation of the request.
    Use this when you know the request_id. To discover request IDs matching criteria,
    use search_requests first.
    """
    if request_id not in BY_ID:
        raise ToolError(f"Request ID '{request_id}' not found.")
    return BY_ID[request_id].as_dict()


if __name__ == "__main__":
    mcp.run()
