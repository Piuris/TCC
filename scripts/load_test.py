"""Teste de carga da API: latência fim a fim sob requisições concorrentes.

Uso (com a API rodando em outro terminal)::

    python scripts/load_test.py --dataset creditcard --requests 2000 --concurrency 16
    python scripts/load_test.py --no-explain        # mede só a predição

Reporta latência p50/p95/p99 (cliente, inclui rede e serialização), a
latência informada pelo servidor e a vazão em transações por segundo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

from fraude.data import load_dataset


async def run(args) -> dict:
    ds = load_dataset(args.dataset)
    rng = np.random.default_rng(args.seed)
    fraud = np.flatnonzero(ds.y.to_numpy() == 1)
    idx = np.concatenate([rng.choice(len(ds.y), args.requests - args.requests // 10),
                          rng.choice(fraud, args.requests // 10)])
    rng.shuffle(idx)
    payloads = [{"transaction_id": f"lt-{i}", **ds.X.iloc[i].to_dict()} for i in idx]
    if args.dataset == "paysim":
        for p in payloads:
            p["step"] = int(p["step"])

    url = f"{args.url}/v1/{ 'paysim' if args.dataset == 'paysim' else 'creditcard'}/predict"
    params = {"explain": str(not args.no_explain).lower()}
    sem = asyncio.Semaphore(args.concurrency)
    client_ms, server_ms, errors = [], [], 0

    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(url, json=payloads[0], params=params)

        async def one(p):
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(url, json=p, params=params)
                client_ms.append((time.perf_counter() - t0) * 1000)
                if r.status_code == 200:
                    server_ms.append(r.json()["latency_ms"])
                else:
                    errors += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(one(p) for p in payloads))
        wall = time.perf_counter() - t0

    c, s = np.asarray(client_ms), np.asarray(server_ms)
    return {
        "requests": len(payloads), "concurrency": args.concurrency, "explain": not args.no_explain,
        "errors": errors, "throughput_tps": round(len(payloads) / wall, 1),
        "client_p50_ms": round(float(np.percentile(c, 50)), 2),
        "client_p95_ms": round(float(np.percentile(c, 95)), 2),
        "client_p99_ms": round(float(np.percentile(c, 99)), 2),
        "server_mean_ms": round(float(s.mean()), 2) if len(s) else None,
        "server_p95_ms": round(float(np.percentile(s, 95)), 2) if len(s) else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--dataset", default="creditcard", choices=["creditcard", "paysim", "synthetic"])
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None, help="Salva o resultado em JSON.")
    args = ap.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
