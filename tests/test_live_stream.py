"""Tests for the live-streaming WebSocket endpoint.

Two layers, on purpose:

* ``TestLiveStreamWebSocket`` uses ``TestClient``. It is fast and it checks the
  message *semantics* of the stream. But TestClient speaks ASGI directly: it
  builds the ``{"type": "websocket"}`` scope itself and never performs an
  HTTP/1.1 Upgrade handshake. It therefore cannot see anything that happens in
  uvicorn's protocol layer, and it will happily pass while the real server
  returns 404 for the very same path.

* ``TestLiveStreamUnderRealUvicorn`` boots the real ``src.api.app`` on a real
  uvicorn server bound to a real TCP port, then connects with a hand-rolled
  RFC6455 client. This is the layer that catches "green tests, 404 in prod".
  It deliberately depends on no websocket client library, so a missing or
  broken server-side WebSocket implementation surfaces as a failed handshake
  here rather than an ImportError at collection time.
"""

import base64
import json
import os
import socket
import struct
import threading
import time

import pytest
from fastapi.testclient import TestClient

from src.api import app


def _collect_ws_messages(client, path="/ws/live?count=10&interval=0.1", max_messages=50):
    """Connect to the websocket and collect all messages until close."""
    messages = []
    with client.websocket_connect(path) as ws:
        while len(messages) < max_messages:
            try:
                data = ws.receive_json()
                messages.append(data)
                if data.get("type") == "stream_end":
                    break
            except Exception:
                break
    return messages


class TestLiveStreamWebSocket:
    """WebSocket live-stream tests using the FastAPI TestClient."""

    def test_messages_arrive_in_order_with_valid_json(self):
        """Messages arrive: stream_start, then processing/incident pairs in
        order, then stream_end. Each is valid parseable JSON."""
        with TestClient(app) as client:
            messages = _collect_ws_messages(client, "/ws/live?count=10&interval=0.1")

        assert len(messages) >= 3, f"Expected at least 3 messages, got {len(messages)}"

        # First message is stream_start
        assert messages[0]["type"] == "stream_start"
        assert messages[0]["total_incidents"] == 10

        # Last message is stream_end
        assert messages[-1]["type"] == "stream_end"
        assert messages[-1]["incidents_sent"] == 10

        # Check incident order: processing messages appear before their
        # corresponding incident messages, and indices are sequential.
        incident_indices = []
        for msg in messages[1:-1]:
            assert msg["type"] in ("processing", "incident"), (
                f"Unexpected type: {msg['type']}"
            )
            if msg["type"] == "incident":
                incident_indices.append(msg["index"])
                # Validate the incident payload has the expected keys
                assert "summary" in msg
                assert "record" in msg
                assert "pipeline_ms" in msg
                s = msg["summary"]
                assert {"incident_id", "cause", "confidence", "status"} <= set(s)

        assert incident_indices == list(range(10)), (
            f"Incidents out of order: {incident_indices}"
        )

    def test_reconnect_does_not_crash_server(self):
        """Disconnecting and reconnecting should work cleanly."""
        with TestClient(app) as client:
            # First connection — receive a few messages then disconnect
            with client.websocket_connect("/ws/live?count=10&interval=0.1") as ws:
                first = ws.receive_json()
                assert first["type"] == "stream_start"
                # Disconnect mid-stream by exiting the context

            # Second connection should also work
            messages = _collect_ws_messages(client, "/ws/live?count=10&interval=0.1")
            assert messages[0]["type"] == "stream_start"
            assert messages[-1]["type"] == "stream_end"

    def test_each_incident_has_full_pipeline_output(self):
        """Each incident message has a full record with all pipeline stages."""
        with TestClient(app) as client:
            messages = _collect_ws_messages(client, "/ws/live?count=10&interval=0.1")

        incident_msgs = [m for m in messages if m["type"] == "incident"]
        assert len(incident_msgs) == 10

        for msg in incident_msgs:
            record = msg["record"]
            for key in (
                "incident_id",
                "detection",
                "correlation",
                "rca_text",
                "impact",
                "recovery",
                "timeline",
                "audit_trail",
            ):
                assert key in record, (
                    f"Missing key {key!r} in {record.get('incident_id', '?')}"
                )

    def test_processing_message_precedes_each_incident(self):
        """For every incident index, a processing message appears before the
        incident message."""
        with TestClient(app) as client:
            messages = _collect_ws_messages(client, "/ws/live?count=10&interval=0.1")

        seen_processing = set()
        for msg in messages:
            if msg["type"] == "processing":
                seen_processing.add(msg["index"])
            elif msg["type"] == "incident":
                assert msg["index"] in seen_processing, (
                    f"Incident index {msg['index']} arrived without a prior processing message"
                )


# ---------------------------------------------------------------------------
# Real-server layer: actual uvicorn, actual TCP socket, actual HTTP Upgrade.
# ---------------------------------------------------------------------------


class WebSocketClosed(Exception):
    """The server sent a close frame or hung up."""


class RawWebSocketClient:
    """A real RFC6455 client over a plain TCP socket.

    Performs the genuine ``GET ... Upgrade: websocket`` handshake, so it
    exercises the same code path a browser does. Uses no websocket library.
    """

    def __init__(self, host, port, path, timeout=90):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._buf = b""
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self._sock.sendall(request.encode())

        # Read the handshake response headers.
        while b"\r\n\r\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise WebSocketClosed("server closed during handshake")
            self._buf += chunk
        head, _, self._buf = self._buf.partition(b"\r\n\r\n")
        self.response_head = head.decode(errors="replace")
        self.status_line = self.response_head.split("\r\n", 1)[0]
        self.status_code = int(self.status_line.split()[1])

    # -- framing ----------------------------------------------------------
    def _read_exact(self, n):
        while len(self._buf) < n:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise WebSocketClosed("connection closed mid-frame")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _send_frame(self, opcode, payload=b""):
        # Client-to-server frames must be masked.
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        header = struct.pack("!BB", 0x80 | opcode, 0x80 | len(payload))
        self._sock.sendall(header + mask + masked)

    def _read_frame(self):
        b0, b1 = self._read_exact(2)
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        length = b1 & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        return fin, opcode, self._read_exact(length)

    def receive_json(self):
        """Return the next JSON text message, reassembling fragments."""
        payload = b""
        opcode = None
        while True:
            fin, op, data = self._read_frame()
            if op == 0x8:
                raise WebSocketClosed("server sent close frame")
            if op == 0x9:  # ping -> pong, keep the connection healthy
                self._send_frame(0xA, data)
                continue
            if op == 0xA:  # stray pong
                continue
            if op != 0x0:
                opcode = op
            payload += data
            if fin:
                break
        assert opcode == 0x1, f"expected a text frame, got opcode {opcode}"
        return json.loads(payload.decode())

    def close(self):
        try:
            self._send_frame(0x8, b"\x03\xe8")
        except OSError:
            pass
        finally:
            self._sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture(scope="module")
def live_server():
    """Serve the real `src.api.app` from a real uvicorn server on a real port.

    This is the whole point of this fixture: `app` is the same object
    `src.run_demo` hands to `uvicorn.run`, served by the same uvicorn protocol
    stack, reached over a real socket.
    """
    import uvicorn

    # Bind first so the port cannot be stolen between choosing and listening.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()

    server = uvicorn.Server(uvicorn.Config(app, log_config=None, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("uvicorn server thread died during startup")
        if time.monotonic() > deadline:
            server.should_exit = True
            raise RuntimeError("uvicorn did not start within 30s")
        time.sleep(0.05)

    yield host, port

    server.should_exit = True
    thread.join(timeout=15)


class TestLiveStreamUnderRealUvicorn:
    """Regression tests for the real serving path (uvicorn + TCP + Upgrade)."""

    def test_uvicorn_resolves_a_websocket_protocol(self):
        """uvicorn must have a WebSocket implementation available.

        With none installed, uvicorn's ws="auto" silently degrades to "none",
        answers the Upgrade as a plain HTTP GET, and every websocket route 404s.
        """
        from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol

        assert AutoWebSocketsProtocol is not None, (
            "no WebSocket library installed (websockets/wsproto): uvicorn would "
            "serve /ws/live as 404. Run: pip install -r requirements.txt"
        )

    def test_real_upgrade_handshake_returns_101(self, live_server):
        """A genuine HTTP Upgrade to /ws/live must be accepted, not 404'd."""
        host, port = live_server
        with RawWebSocketClient(host, port, "/ws/live?count=10&interval=0.05") as ws:
            assert ws.status_code == 101, (
                f"/ws/live did not upgrade under real uvicorn: {ws.status_line!r}\n"
                f"{ws.response_head}"
            )

    def test_real_server_streams_at_least_three_incidents(self, live_server):
        """Over a real socket: stream_start, ordered incidents, stream_end."""
        host, port = live_server
        messages = []
        with RawWebSocketClient(host, port, "/ws/live?count=10&interval=0.05") as ws:
            assert ws.status_code == 101, f"handshake failed: {ws.status_line!r}"
            while len(messages) < 60:
                try:
                    msg = ws.receive_json()
                except WebSocketClosed:
                    break
                messages.append(msg)
                if msg.get("type") == "stream_end":
                    break

        assert messages, "no messages received over the real websocket"
        assert messages[0]["type"] == "stream_start"
        assert messages[-1]["type"] == "stream_end"

        incidents = [m for m in messages if m["type"] == "incident"]
        assert len(incidents) >= 3, (
            f"expected at least 3 streamed incidents, got {len(incidents)}"
        )
        assert [m["index"] for m in incidents] == list(range(len(incidents)))
        for msg in incidents:
            assert {"incident_id", "cause", "confidence", "status"} <= set(msg["summary"])
            assert "record" in msg

    def test_non_websocket_routes_still_work_on_the_real_server(self, live_server):
        """Sanity check that the fixture really is serving the demo app."""
        import httpx

        host, port = live_server
        response = httpx.get(f"http://{host}:{port}/api/health", timeout=10)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
