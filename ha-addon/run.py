# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import json, os, shlex, sys

OPTIONS = "/data/options.json"
SERVER  = "/app/src/run.py"

def main():
    try:
        with open(OPTIONS, "r", encoding="utf-8") as f:
            o = json.load(f)
    except Exception as e:
        print(f"[addon] options load error: {e}", flush=True)
        o = {}

    host  = o.get("host", "0.0.0.0")
    port  = int(o.get("port", 8788))
    cfg   = o.get("config_path") or ""
    extra = o.get("extra_args", "")

    cmd = [sys.executable, SERVER, "--host", str(host), "--port", str(port)]
    if cfg:
        cmd += ["--config", cfg]
    if extra.strip():
        cmd += shlex.split(extra)

    print("[addon] exec:", " ".join(shlex.quote(c) for c in cmd), flush=True)
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    main()
