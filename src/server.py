# SPDX-License-Identifier: AGPL-3.0-only
# 🄯 2022, Alexey Parfenov <zxed@alkatrazstudio.net>

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Final, Optional, Type

from ai import GeneratedOutput
from logits_warper_override import WarperId
from rep_pen_processor import RepPenGenerated


class RequestData:
    prompt: str
    preamble: str
    generated_tokens_count: int
    max_total_tokens: int
    sequences_count: int
    temperature: Optional[float]
    top_k: Optional[int]
    top_p: Optional[float]
    tfs: Optional[float]
    typical: Optional[float]
    top_a: Optional[float]
    warpers_order: list[WarperId]
    repetition_penalty: Optional[float]
    repetition_penalty_range: Optional[int]
    repetition_penalty_slope: Optional[float]
    repetition_penalty_include_preamble: bool
    repetition_penalty_include_generated: RepPenGenerated
    repetition_penalty_truncate_to_input: bool
    repetition_penalty_prompt: Optional[str]
    stop_strings: list[str]
    truncate_prompt_until: list[str]

    def __init__(self, data: dict):
        self.prompt = str(data["prompt"]) if "prompt" in data else ""
        self.preamble = str(data["preamble"]) if "preamble" in data else ""
        self.generated_tokens_count = int(data["generated_tokens_count"])
        self.max_total_tokens = int(data["max_total_tokens"])
        self.sequences_count = int(data["sequences_count"]) if "sequences_count" in data else 1
        self.temperature = RequestData.get_val(data, "temperature", float)
        self.top_k = RequestData.get_val(data, "top_k", int)
        self.top_p = RequestData.get_val(data, "top_p", float)
        self.tfs = RequestData.get_val(data, "tfs", float)
        self.typical = RequestData.get_val(data, "typical", float)
        self.top_a = RequestData.get_val(data, "top_a", float)
        self.warpers_order = [WarperId(x) for x in data["warpers_order"]] if "warpers_order" in data else []
        self.repetition_penalty = RequestData.get_val(data, "repetition_penalty", float)
        self.repetition_penalty_range = RequestData.get_val(data, "repetition_penalty_range", int)
        self.repetition_penalty_slope = RequestData.get_val(data, "repetition_penalty_slope", float)
        self.repetition_penalty_include_preamble = RequestData.get_bool(data, "repetition_penalty_include_preamble")
        self.repetition_penalty_truncate_to_input = RequestData.get_bool(data, "repetition_penalty_truncate_to_input")
        self.repetition_penalty_include_generated = \
            RepPenGenerated(str(data["repetition_penalty_include_generated"])) \
            if "repetition_penalty_include_generated" in data \
            else RepPenGenerated.SLIDE
        self.repetition_penalty_prompt = RequestData.get_val(data, "repetition_penalty_prompt", str)
        self.stop_strings = RequestData.get_vals(data, "stop_strings", str)
        self.truncate_prompt_until = RequestData.get_vals(data, "truncate_prompt_until", str)

        if self.top_p == 0:
            self.top_p = None
        if self.top_k == 0:
            self.top_k = None
        if self.tfs == 0:
            self.tfs = None
        if self.typical == 0:
            self.typical = None
        if self.top_a == 0:
            self.top_a = None
        if self.temperature == 0:
            self.temperature = None
        if self.repetition_penalty == 0:
            self.repetition_penalty = None
        if self.repetition_penalty_range is None:
            self.repetition_penalty_range = 0
        if self.repetition_penalty_slope == 0:
            self.repetition_penalty_slope = None
        if self.repetition_penalty_prompt == "":
            self.repetition_penalty_prompt = None

    @staticmethod
    def get_val(data: dict, key: str, t: Type) -> Optional[Any]:
        if key not in data:
            return None
        if data[key] is None:
            return None
        return t(data[key])

    @staticmethod
    def get_vals(data: dict, key: str, t: Type) -> list:
        if key not in data:
            return []
        if data[key] is None:
            return []
        return [t(v) for v in data[key]]

    @staticmethod
    def get_bool(data: dict, key: str) -> bool:
        if key not in data:
            return False
        return bool(data[key])


Callback = Callable[[RequestData], GeneratedOutput]

ENDPOINT_PATH: Final[str] = "/generate"
SERVER_NAME: Final[str] = "Neodim Server"
SERVER_VERSION: Final[str] = "0.7"


def name_and_version():
    return f"{SERVER_NAME} v{SERVER_VERSION}"


def handler_with_callback(callback: Callback) -> Type[BaseHTTPRequestHandler]:
    class HttpServerHandler(BaseHTTPRequestHandler):
        def load_json(self) -> dict:
            content_len = int(self.headers["content-length"])
            json_str = self.rfile.read(content_len)

            in_dict = json.loads(json_str)
            return in_dict

        def generate(self) -> dict:
            in_dict = self.load_json()
            in_data = RequestData(in_dict)

            out_data = callback(in_data)
            out_dict = out_data.to_dict()
            return out_dict

        def do_POST(self) -> None:
            try:
                if self.path == ENDPOINT_PATH:
                    out_dict = self.generate()
                else:
                    raise RuntimeError(f"Unsupported URI: {self.path}")
                out_code = 200
            except Exception as e:
                out_code = 500
                out_dict = {
                    "error": f"{e.__class__.__name__}: {str(e)}"
                }

            self.send_response(out_code)
            if "origin" in self.headers:
                self.send_header("access-control-allow-origin", self.headers["origin"])

            out_json = json.dumps(out_dict).encode()
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
