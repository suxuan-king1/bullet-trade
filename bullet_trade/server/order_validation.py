import math


def normalize_order(payload):
    """让风控、幂等指纹与执行共享无冲突的数量、方向和价格。"""
    security = payload.get("security")
    if not isinstance(security, str) or not security.strip():
        raise ValueError("缺少 security")
    style = payload.get("style") or {}
    if not isinstance(style, dict):
        raise ValueError("style 必须为对象")
    style = dict(style)
    side = str(payload.get("side") or "").strip().upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("side 必须为 BUY 或 SELL")
    amount = float(payload.get("amount", payload.get("volume", 0)))
    if not math.isfinite(amount) or amount <= 0 or not amount.is_integer():
        raise ValueError("amount 必须为正整数")
    if "amount" in payload and "volume" in payload and float(payload["volume"]) != amount:
        raise ValueError("amount 与 volume 冲突")
    prices = [payload.get("price"), style.get("price"), style.get("protect_price"), style.get("limit_price")]
    prices = [float(value) for value in prices if value is not None]
    if any(not math.isfinite(value) or value <= 0 for value in prices):
        raise ValueError("价格必须为有限正数")
    if prices and any(value != prices[0] for value in prices):
        raise ValueError("price 与 style 价格冲突")
    native_types = [payload.get("pr_type"), payload.get("prType"), style.get("pr_type"), style.get("prType")]
    native_types = [float(value) for value in native_types if value is not None]
    if any(not math.isfinite(value) or not value.is_integer() for value in native_types):
        raise ValueError("pr_type 必须为整数")
    native_types = [int(value) for value in native_types]
    if native_types and len(set(native_types)) != 1:
        raise ValueError("pr_type 冲突")
    if native_types:
        payload["pr_type"] = native_types[0]
    if prices:
        payload["price"] = prices[0]
        field = "protect_price" if str(style.get("type", "")).lower() == "market" else "price"
        style[field] = prices[0]
    payload.update(side=side, amount=int(amount), style=style)
