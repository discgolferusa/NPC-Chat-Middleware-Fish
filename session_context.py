"""
Request-scoped campaign session identity.

Every concurrent table is identified by (owner_user_id, campaign_id).
Globals are not used for live play state.
"""

from __future__ import annotations

from contextvars import ContextVar

_REQUEST_OWNER: ContextVar[str] = ContextVar("session_owner", default="")
_REQUEST_CAMPAIGN: ContextVar[str] = ContextVar("session_campaign", default="")


def set_request_owner(owner_user_id: str | None) -> None:
    _REQUEST_OWNER.set((owner_user_id or "").strip())


def set_request_campaign(campaign_id: str | None) -> None:
    _REQUEST_CAMPAIGN.set((campaign_id or "").strip())


def set_request_context(owner_user_id: str | None = None, campaign_id: str | None = None) -> None:
    if owner_user_id is not None:
        set_request_owner(owner_user_id)
    if campaign_id is not None:
        set_request_campaign(campaign_id)


def get_request_owner() -> str:
    return (_REQUEST_OWNER.get() or "").strip()


def get_request_campaign() -> str:
    return (_REQUEST_CAMPAIGN.get() or "").strip()


def clear_request_context() -> None:
    _REQUEST_OWNER.set("")
    _REQUEST_CAMPAIGN.set("")


def session_key(owner_user_id: str | None = None, campaign_id: str | None = None) -> str | None:
    owner = (owner_user_id if owner_user_id is not None else get_request_owner()).strip()
    campaign = (campaign_id if campaign_id is not None else get_request_campaign()).strip()
    if not owner or not campaign:
        return None
    return f"{owner.casefold()}\x1f{campaign.casefold()}"
