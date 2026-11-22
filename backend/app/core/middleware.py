"""Small ASGI guards for request sizes and response security headers."""

from __future__ import annotations

import json


class RequestBodyTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    def __init__(self, app, *, upload_limit: int, json_limit: int = 64 * 1024):
        self.app = app
        self.upload_limit = upload_limit
        self.json_limit = json_limit

    def _limit(self, path: str) -> int:
        if path in {"/api/v1/analyze", "/api/v1/provenance"}:
            return self.upload_limit + 128 * 1024
        return self.json_limit

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or not scope.get("path", "").startswith("/api/"):
            return await self.app(scope, receive, send)
        limit = self._limit(scope.get("path", ""))
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > limit:
                    return await self._send_too_large(scope, send)
            except ValueError:
                return await self._send_too_large(scope, send)

        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message.get("type") == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    raise RequestBodyTooLarge
            return message

        response_started = False

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            if not response_started:
                await self._send_too_large(scope, send)

    @staticmethod
    async def _send_too_large(scope, send):
        body = json.dumps({"detail": "Request body exceeds the configured size limit."}).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})


def apply_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cache-Control", "no-store")
    return response
