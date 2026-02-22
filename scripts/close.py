from __future__ import annotations

import argparse
import sys
from typing import Any

from scalp.config import Settings
from scalp.exchange.okx_rest import OKXRestClient


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _signed_position(pos: dict[str, Any]) -> float:
    qty = _as_float(pos.get("pos"), 0.0)
    pos_side = str(pos.get("posSide") or "").lower()
    if pos_side == "short":
        return -abs(qty)
    if pos_side == "long":
        return abs(qty)
    return qty


def _build_close_order(pos: dict[str, Any]) -> dict[str, str] | None:
    inst_id = str(pos.get("instId") or "")
    if not inst_id:
        return None

    signed_qty = _signed_position(pos)
    if abs(signed_qty) <= 0:
        return None

    side = "sell" if signed_qty > 0 else "buy"
    qty = abs(signed_qty)
    td_mode = str(pos.get("mgnMode") or "cross")
    pos_side = str(pos.get("posSide") or "").lower()

    order: dict[str, str] = {
        "instId": inst_id,
        "tdMode": td_mode,
        "side": side,
        "ordType": "market",
        "sz": f"{qty:.8f}",
        "reduceOnly": "true",
    }
    if pos_side in {"long", "short"}:
        order["posSide"] = pos_side
    return order


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Immediately close all open OKX positions using env credentials."
    )
    parser.add_argument(
        "--env-file",
        default="demo.env",
        help="Path to env file with OKX credentials (default: demo.env).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show close orders without sending them.",
    )
    args = parser.parse_args()

    settings = Settings(_env_file=args.env_file)
    if not settings.okx_api_key or not settings.okx_api_secret or not settings.okx_passphrase:
        print("Missing OKX credentials in env file.", file=sys.stderr)
        return 2

    rest = OKXRestClient(settings)
    positions = rest.get_positions() or []
    close_orders = [o for o in (_build_close_order(p) for p in positions) if o]

    print(
        f"Found {len(close_orders)} open position(s) "
        f"(simulated={settings.okx_simulated})."
    )
    if not close_orders:
        print("Nothing to close.")
        return 0

    for idx, order in enumerate(close_orders, start=1):
        inst_id = order["instId"]
        side = order["side"]
        size = order["sz"]
        td_mode = order["tdMode"]
        pos_side = order.get("posSide", "net")
        print(f"[{idx}/{len(close_orders)}] {side} {size} {inst_id} tdMode={td_mode} posSide={pos_side}")
        if args.dry_run:
            continue

        # Use raw trade endpoint to pass reduceOnly/posSide explicitly.
        result = rest._trade.place_order(**order)  # noqa: SLF001
        data = (result or {}).get("data") or [{}]
        item = data[0] if data else {}
        if item.get("sCode") != "0":
            print(f"  FAILED: {item.get('sMsg')}", file=sys.stderr)
        else:
            print(f"  OK: ordId={item.get('ordId', '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
