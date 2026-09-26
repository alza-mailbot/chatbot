"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Report service liveness.

    Returns:
        dict[str, str]: A static status payload.
    """
    return {"status": "ok"}
