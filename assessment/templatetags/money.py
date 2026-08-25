"""Template filters for money.

`rupees` calls `core.cost.money`, the same function the admin uses, so the web
page, the admin changelist and the cost model can never disagree about how a
figure is written. Indian digit grouping (12,34,567 rather than 1,234,567) is not
something Django's number formatting or humanize's intcomma will do, and getting
it wrong on an Indian repair estimate looks careless.
"""

from __future__ import annotations

from django import template

from core.cost import money

register = template.Library()


@register.filter(name="rupees")
def rupees(value) -> str:
    """Format a Decimal or float as ₹12,34,567.

    Returns an em dash for None rather than "₹0": a missing figure and a figure
    of zero mean different things on a quotation.
    """
    if value is None:
        return "—"
    try:
        return money(value)
    except (TypeError, ValueError):
        return "—"
