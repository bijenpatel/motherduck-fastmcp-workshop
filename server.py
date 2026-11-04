"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Literal

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import InputRequiredResult

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, closest
from sf311.kit import ask_choice, caller_email, read_choice

logger = logging.getLogger(__name__)

mcp = FastMCP("sf-311")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10
HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")


def mask_address(address: str) -> str:
    match = HOUSE_NUMBER.match(address)
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def _format_request(request) -> dict:
    row = request.as_dict()
    row["address"] = mask_address(row["address"])
    return row

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
    Street addresses are masked to the block.
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
    return [_format_request(r) for r in rows[:limit]]


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

    Street addresses are masked to the block. Logs the caller's email.
    Use this when you know the request_id. To discover request IDs matching criteria,
    use search_requests first.
    """
    if request_id not in BY_ID:
        raise ToolError(f"Request ID '{request_id}' not found.")
    caller = caller_email() or "local"
    logger.info("get_request %s for %s", request_id, caller)
    return _format_request(BY_ID[request_id])


@mcp.tool(annotations={"readOnlyHint": True})
def longest_close_time(
    service: str | None = None,
    neighborhood: str | None = None,
) -> dict:
    """Find the closed 311 request that took the longest time from opened to closed.

    Returns the maximum resolution time in whole hours (rounded) and decimal hours,
    along with the request ID, service, neighborhood, and masked request details.
    Use this for questions about the longest, slowest, or maximum time taken to resolve a request.
    For median resolution times across groups, use median_close_hours instead.
    For counting requests, use count_requests instead.
    """
    _validate_service(service)
    _validate_neighborhood(neighborhood)
    rows = [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and r.hours_to_close is not None
    ]
    if not rows:
        raise ToolError("No closed requests found matching the given filters.")

    slowest = max(rows, key=lambda r: r.hours_to_close)
    return {
        "whole_hours": round(slowest.hours_to_close),
        "longest_hours": round(slowest.hours_to_close, 1),
        "request_id": slowest.request_id,
        "service": slowest.service,
        "neighborhood": slowest.neighborhood,
        "request": _format_request(slowest),
    }


def _official_parts(local_name: str) -> list[str]:
    """The official neighborhoods a local name overlaps, largest first."""
    counts = Counter(r.neighborhood for r in REQUESTS if r.local_name == local_name and r.neighborhood)
    return [name for name, _ in counts.most_common()]


@mcp.tool(annotations={"readOnlyHint": True})
def count_in_neighborhood(
    name: str,
    ctx: Context,
    boundary: str | None = None,
) -> int | str | InputRequiredResult:
    """Count 311 requests in a neighborhood, by its official name or the name people use.

    A local name such as "Lower Nob Hill" can span several official
    neighborhoods. Then this tool asks the user which part to count. If your
    client cannot show forms, pass boundary instead: one of the official
    neighborhoods the error or question names, or "all".
    """
    if name in NEIGHBORHOODS:
        return sum(1 for r in REQUESTS if r.neighborhood == name)
    if name not in LOCAL_NAMES:
        suggestions = ", ".join(closest(name, NEIGHBORHOODS + LOCAL_NAMES)) or "call dataset_info"
        raise ToolError(f"Unknown neighborhood '{name}'. Did you mean: {suggestions}?")

    parts = _official_parts(name)
    everything = f"All of {name}"
    choice = read_choice(ctx, "boundary")
    if choice is not None:
        if not choice.accepted:
            return "No neighborhood chosen, so nothing was counted."
        boundary = choice.value
    if boundary is None and len(parts) > 1:
        return ask_choice(
            "boundary",
            f"{name} spans {', '.join(parts)}. Which part should I count?",
            [*parts, everything],
        )

    in_area = [r for r in REQUESTS if r.local_name == name]
    if boundary in (None, "all", everything):
        return len(in_area)
    if boundary not in parts:
        raise ToolError(f"{name} does not overlap '{boundary}'. Choose one of: {', '.join(parts)}, or all.")
    return sum(1 for r in in_area if r.neighborhood == boundary)


if __name__ == "__main__":
    mcp.run()
