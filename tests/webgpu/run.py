"""Serve and run standalone WebGPU numerical tests in Chrome or Edge."""
import argparse
import functools
import http.server
import json
import pathlib
import subprocess
import threading
import time
import urllib.parse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--executable", default="FluidSimGPUTests")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    test_args = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    build = pathlib.Path(args.build).resolve()
    output = pathlib.Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    shell = pathlib.Path(__file__).with_name("index.html").read_text(encoding="utf-8")
    shell = shell.replace('src="FluidSimGPUTests.js"', 'src="' + urllib.parse.quote(args.executable, safe="") + '.js"').encode("utf-8")
    completed = threading.Event()
    result = {"browser": args.browser, "build": str(build), "executable": args.executable, "arguments": test_args, "result": "timeout"}
    test_log = output.with_suffix(".stdout.log").open("w", encoding="utf-8")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            super().end_headers()

        def log_message(self, *args):
            pass

        def do_GET(self):
            if urllib.parse.urlparse(self.path).path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(shell)
            else:
                super().do_GET()

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
            if path == "/log":
                test_log.write(body + "\n")
                test_log.flush()
            elif path == "/result":
                result.update(json.loads(body))
                result["result"] = "pass" if result["code"] == 0 else "fail"
                completed.set()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(build)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = build / ("browser-profile-" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(time.monotonic_ns()))
    query = urllib.parse.urlencode([("arg", value) for value in test_args])
    arguments = [args.browser, "--headless=new", "--no-first-run", "--no-default-browser-check",
                 "--enable-logging=stderr", "--user-data-dir=" + str(profile), f"http://127.0.0.1:{server.server_port}/?{query}"]
    started = time.monotonic()
    try:
        with output.with_suffix(".browser.log").open("w") as log:
            process = subprocess.Popen(arguments, stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                completed.wait(args.timeout)
            finally:
                process.terminate()
                process.wait(timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        test_log.close()
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "lines"}))
    return 0 if result["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
