"""python -m ticketpilot.ambient <command>   (spec-a7)

  replay STREAM [--offline] [--obedient]   run a stream through the pipeline on event time
  serve [--port N]                          webhook intake plus a background worker (needs the mock API on :8000 and an LLM key)
  approvals list|approve|edit|reject        the human side of the approval queue
  verify-audit                              check the audit log's hash chain
  detect STREAM                             run the anomaly detector alone
State lives in runs/ambient/ (override with TICKETPILOT_AMBIENT_DIR)."""
import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from ticketpilot.ambient import anomaly, offline
from ticketpilot.ambient.approvals import Approvals, ApprovalError
from ticketpilot.ambient.audit import AuditLog
from ticketpilot.ambient.events import load_stream
from ticketpilot.ambient.replay import inprocess_toolbox, replay
from ticketpilot.ambient.store import Store, default_path


def _state(args):
    base = Path(args.dir) if args.dir else default_path()
    return Store(base / "ambient.db"), AuditLog(base / "audit.jsonl")


async def _replay(args) -> int:
    store, audit = _state(args)
    events = load_stream(Path(args.stream))[: args.limit]
    from ticketpilot.mcp_client import open_toolbox
    box = inprocess_toolbox if args.offline else open_toolbox
    async with box() as tb:
        ctx = offline.patched(args.obedient) if args.offline else None
        if ctx:
            ctx.__enter__()
        try:
            r = await replay(events, store, audit, tb)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
    print(json.dumps({"events": len(events), "done": store.count("done"), "duplicates": store.count("duplicate"), "shed": store.count("shed"),
                      "alerts": len(store.alerts()), "pending_approvals": len(store.approvals("pending"))}, indent=1))
    for a in store.alerts():
        print("ALERT:", a["text"])
    return 0


async def _serve(args) -> int:
    import uvicorn
    from ticketpilot.ambient.triage import Triage, drain
    from ticketpilot.ambient.webhook import create_app
    from ticketpilot.mcp_client import open_toolbox
    store, audit = _state(args)
    app = create_app(store, audit)
    async with open_toolbox() as tb:
        triage = Triage(store, audit, tb)

        async def worker():
            while True:
                await drain(store, triage)
                await asyncio.sleep(1.0)

        task = asyncio.create_task(worker())
        server = uvicorn.Server(uvicorn.Config(app, port=args.port, log_level="warning"))
        try:
            await server.serve()
        finally:
            task.cancel()
    return 0


def _approvals(args) -> int:
    store, audit = _state(args)
    ap = Approvals(store, audit)
    try:
        if args.action == "list":
            for i in store.approvals(args.status):
                p = i["payload"]
                print(f"#{i['id']:<4} {i['status']:<9} {'URGENT ' if i['urgent'] else ''}{i['kind']:<18} {i['event_id']} -> {i['recipient']}: "
                      f"{(p.get('reply') or p.get('reason') or '')[:90]!r}")
        elif args.action == "approve":
            ap.approve(args.id, args.by)
        elif args.action == "edit":
            ap.edit(args.id, args.by, args.text)
        elif args.action == "reject":
            ap.reject(args.id, args.by, args.reason or "")
        if args.action != "list":
            print(f"approval {args.id}: {store.approval(args.id)['status']}")
    except ApprovalError as e:
        print(f"refused: {e}")
        return 1
    return 0


def main(argv=None) -> int:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="ticketpilot.ambient", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", help="state directory (default runs/ambient)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay"); r.add_argument("stream"); r.add_argument("--offline", action="store_true")
    r.add_argument("--obedient", action="store_true"); r.add_argument("--limit", type=int, default=None)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8100)
    a = sub.add_parser("approvals"); a.add_argument("action", choices=["list", "approve", "edit", "reject"])
    a.add_argument("id", nargs="?", type=int); a.add_argument("--by", default="reviewer"); a.add_argument("--text"); a.add_argument("--reason")
    a.add_argument("--status", choices=["pending", "approved", "rejected"])
    sub.add_parser("verify-audit")
    d = sub.add_parser("detect"); d.add_argument("stream"); d.add_argument("--k", type=float, default=anomaly.Params().k)
    args = ap.parse_args(argv)
    if args.cmd == "replay":
        return asyncio.run(_replay(args))
    if args.cmd == "serve":
        return asyncio.run(_serve(args))
    if args.cmd == "approvals":
        return _approvals(args)
    if args.cmd == "verify-audit":
        ok, msg = _state(args)[1].verify()
        print("audit chain:", msg)
        return 0 if ok else 1
    if args.cmd == "detect":
        ev = load_stream(Path(args.stream))
        start = (min(e.epoch for e in ev) // anomaly.BUCKET_S) * anomaly.BUCKET_S
        pts = [(e.epoch, e.event_id, "unclassified") for e in ev]
        for al in anomaly.detect(pts, start, max(e.epoch for e in ev) + 1, anomaly.Params(k=args.k)):
            print(al.text)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
