# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import sys
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Final

import jsons
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from ai import GeneratedOutput, RequestData


@dataclass
class ServerRequestData(RequestData):
    required_server_version: str | None = None


Callback = Callable[[ServerRequestData], GeneratedOutput]


ENDPOINT_PATH: Final[str] = "/generate"
SERVER_NAME: Final[str] = "Neodim Server"
SERVER_VERSION: Final[Version] = Version("0.11")


def name_and_version():
    return f"{SERVER_NAME} v{SERVER_VERSION}"


def handler_with_callback(callback: Callback) -> type[BaseHTTPRequestHandler]:
    class HttpServerHandler(BaseHTTPRequestHandler):
        def generate(self) -> GeneratedOutput:
            content_len = int(self.headers["content-length"])
            json_str = self.rfile.read(content_len)
            in_data = jsons.loadb(
                json_str,
                ServerRequestData,
                strict=True
            )

            if in_data.required_server_version is not None:
                specifier = SpecifierSet(in_data.required_server_version)
                if SERVER_VERSION not in specifier:
                    raise RuntimeError(
                        f"Requested server version: {specifier}, current server version: {SERVER_VERSION}")

            out_data = callback(in_data)
            return out_data

        def do_POST(self) -> None:
            try:
                if self.path == ENDPOINT_PATH:
                    out_data = self.generate()
                else:
                    raise RuntimeError(f"Unsupported URI: {self.path}")
                out_code = 200
            except Exception as e:
                out_code = 500
                out_data = {
                    "error": f"{e.__class__.__name__}: {str(e)}"
                }

            self.send_response(out_code)
            if "origin" in self.headers:
                self.send_header("access-control-allow-origin", self.headers["origin"])

            out_json = jsons.dumpb(out_data)
            self.send_header("content-type", "application/json")
            self.send_header("content-size", str(len(out_json)))
            self.end_headers()
            self.wfile.write(out_json)

        def do_OPTIONS(self) -> None:
            self.send_response(200)
            if "origin" in self.headers:
                self.send_header("access-control-allow-origin", self.headers["origin"])
            self.send_header("access-control-allow-methods", "POST, OPTIONS")
            if "access-control-request-headers" in self.headers:
                self.send_header("access-control-allow-headers", self.headers["access-control-request-headers"])
            self.send_header("access-control-allow-credentials", "true")
            self.end_headers()

        def version_string(self) -> str:
            return name_and_version()

    return HttpServerHandler


def run(ip: str, port: int, callback: Callback) -> None:
    server_address = (ip, port)
    handler = handler_with_callback(callback)
    httpd = HTTPServer(server_address, handler)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\r  \b\b")  # hide ^C
