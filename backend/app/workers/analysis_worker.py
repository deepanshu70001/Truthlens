"""Run the durable TruthLens analysis worker as a separate process.

Start after applying migrations with:
    python -m backend.app.workers.analysis_worker
"""

from __future__ import annotations

import asyncio
import signal

from backend.app.config import settings
from backend.app.core.logging import setup_logging
from backend.app.db.session import async_session_factory, engine
from backend.app.services.job_queue import run_worker


async def main() -> None:
    settings.validate_runtime_security()
    setup_logging()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        await run_worker(async_session_factory, stop_event=stop_event)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
