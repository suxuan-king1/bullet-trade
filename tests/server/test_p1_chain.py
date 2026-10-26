import asyncio
import copy
import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from bullet_trade.core.risk_control import RiskController
from bullet_trade.server.adapters.base import AccountRouter, AdapterBundle
from bullet_trade.server.adapters.big_qmt import BigQmtBrokerAdapter, BigQmtGatewayClient, BigQmtGatewayConfig
from bullet_trade.server.app import ServerApplication
from bullet_trade.server.config import AccountConfig, ServerConfig, _parse_allowlist
from bullet_trade.server.order_validation import normalize_order
from helpers import bullet_trade_jq_remote_helper as helper


def load_gateway():
    path = Path(__file__).parents[2] / "helpers" / "big_qmt_gateway_strategy_sample.py"
    spec = importlib.util.spec_from_file_location("p1_gateway", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    module = load_gateway()
    module.SUBMISSION_STORE_PATH = str(tmp_path / "submissions.sqlite3")
    state = SimpleNamespace(orders=[], positions=[], calls=[], crash=False)
    monkeypatch.setattr(module, "_query_positions", lambda *args: copy.deepcopy(state.positions))
    monkeypatch.setattr(module, "_query_orders", lambda *args: copy.deepcopy(state.orders))

    def native(context, request):
        state.calls.append(copy.deepcopy(request))
        if state.crash:
            raise RuntimeError("simulated crash after broker call")
        state.orders.append({
            "order_id": str(len(state.calls)), "security": request["security"],
            "amount": request["amount"], "filled": 0, "raw_status": 50,
            "side": request["side"], "price": request["price"], "order_price": request["price"],
            "qmt_user_order_id": request["qmt_user_order_id"], "order_time": time.time(),
            "sub_account_id": request.get("sub_account_id", ""),
        })
        return module._ok({"order_id": "", "status": "submit_unknown"})

    monkeypatch.setattr(module, "_place_order_native", native)
    return module, state


def request(key="signal-1", **values):
    result = {"account_id": "test-account", "account_type": "stock", "security": "000001.XSHE",
              "side": "BUY", "amount": 100, "price": 10.0, "style": {"type": "limit", "price": 10.0},
              "pr_type": 11, "idempotency_key": key, "qmt_user_order_id": "BT-" + key, "wait_timeout": 0.1}
    result.update(values)
    return result


def test_durable_duplicate_and_conflict(gateway):
    module, state = gateway
    module._place_order(object(), request())
    response = module._place_order(object(), request())
    assert response["value"]["order_id"] == "1"
    assert len(state.calls) == 1
    assert module._place_order(object(), request(amount=200))["code"] == "IDEMPOTENCY_CONFLICT"
    assert len(state.calls) == 1


def test_gateway_restart_and_lost_result(gateway):
    module, state = gateway
    state.crash = True
    with pytest.raises(RuntimeError):
        module._place_order(object(), request())
    restarted = load_gateway()
    restarted.SUBMISSION_STORE_PATH = module.SUBMISSION_STORE_PATH
    restarted._query_orders = module._query_orders
    restarted._query_positions = module._query_positions
    restarted._place_order_native = module._place_order_native
    response = restarted._place_order(object(), request())
    assert response["value"]["submit_unknown"] is True
    assert len(state.calls) == 1


def test_concurrent_duplicates_execute_once(gateway):
    module, state = gateway
    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(lambda unused: module._place_order(object(), request()), range(8)))
    assert all(response["ok"] for response in responses)
    assert len(state.calls) == 1


def test_target_blocks_new_keys_until_position_settles(gateway):
    module, state = gateway
    module._place_order(object(), request(target_amount=100))
    blocked = module._place_order(object(), request("signal-2", target_amount=100))
    assert blocked["ok"] is False
    state.orders[0].update(raw_status=56, filled=100)
    assert module._place_order(object(), request("signal-2", target_amount=100))["ok"] is False
    state.positions = [{"security": "000001.XSHE", "amount": 100}]
    response = module._place_order(object(), request("signal-2", target_amount=100))
    assert response["value"]["status"] == "no_op"
    assert len(state.calls) == 1


def test_type_error_never_retries_native_signature():
    module = load_gateway()
    calls = []

    def native(*args):
        calls.append(args)
        raise TypeError("failure after side effect")

    with pytest.raises(TypeError):
        module._call_passorder(native, 23, "account", "000001.SZ", 11, 10, 100, "qmt", "BT-1", object(), 2)
    assert len(calls) == 1


def test_cancel_without_owner_evidence_is_rejected(gateway, monkeypatch):
    module, state = gateway
    monkeypatch.setattr(module, "_qmt_global", lambda name: pytest.fail("must not call cancel"))
    result = module._cancel_order(object(), dict(request(), sub_account_id="alpha", order_id="external"))
    assert result["code"] == "VIRTUAL_ACCOUNT_MISMATCH"


@pytest.mark.parametrize("changes", [{"price": 100}, {"side": "BUI"}, {"amount": 1.2}, {"amount": float("nan")}, {"price": float("inf")}])
def test_invalid_order_rejected_before_broker(changes):
    with pytest.raises(ValueError):
        normalize_order(request(**changes))


def test_unknown_account_rejected():
    router = AccountRouter([AccountConfig(key="main", account_id="account")])
    assert router.get(None).config.key == "main"
    with pytest.raises(KeyError):
        router.get("maim")


@pytest.mark.parametrize("value", ["bad-ip", "127.0.0.1,bad-ip", "", ";"])
def test_invalid_allowlist_fails_closed(value):
    with pytest.raises(ValueError):
        _parse_allowlist(value)


def test_existing_and_pending_holdings_count_toward_limit():
    risk = RiskController()
    risk.config.update(max_position_ratio=50, max_order_value=1000000, max_daily_trade_value=10000000)
    with pytest.raises(ValueError, match="仓位"):
        risk.check_order(20000, 1, "000001.XSHE", 100000, current_position_value=40000)
    with pytest.raises(ValueError, match="仓位"):
        risk.check_order(10000, 1, "000001.XSHE", 100000, current_position_value=35000, pending_buy_value=10000)


class LocalGatewayClient:
    def __init__(self, module):
        self.module = module
        self.decoder = BigQmtGatewayClient(BigQmtGatewayConfig())

    async def health(self):
        return {"submission_protocol": "durable-submissions/v1"}

    async def post(self, path, payload):
        response = self.module._dispatch_qmt_action(object(), path.strip("/"), copy.deepcopy(payload))
        return self.decoder._unwrap_response(response)

    async def post_first(self, paths, payload):
        return {"ticks": {payload["security"]: {"lastPrice": 10, "time": int(time.time() * 1000)}}}


def make_app(module):
    config = ServerConfig(token="token", accounts=[AccountConfig(key="default", account_id="test-account")], enable_broker=True)
    router = AccountRouter(config.accounts)
    adapter = BigQmtBrokerAdapter(config, router, LocalGatewayClient(module))
    return ServerApplication(config, router, AdapterBundle(None, adapter)), adapter


def test_server_restart_reuses_gateway_order(gateway):
    module, state = gateway

    async def run():
        session = SimpleNamespace(account_key="default", sub_account_id=None)
        app, adapter = make_app(module)
        original = request()
        first = await app._dispatch_broker(session, "place_order", copy.deepcopy(original))
        restarted, adapter = make_app(module)
        second = await restarted._dispatch_broker(session, "place_order", copy.deepcopy(original))
        resolved = await restarted._dispatch_broker(session, "resolve_submission", {
            "write_action": "broker.place_order", "idempotency_key": original["idempotency_key"], "request_payload": original,
        })
        assert first["order_id"] == second["order_id"] == resolved["order_id"] == "1"
        assert len(state.calls) == 1

    asyncio.run(run())


def test_default_market_gets_protection_price(gateway):
    module, state = gateway

    async def run():
        app, adapter = make_app(module)
        payload = request(style={"type": "market"}, price=None)
        payload.pop("pr_type")
        result = await app._dispatch_broker(SimpleNamespace(account_key="default", sub_account_id=None), "place_order", payload)
        assert result["order_id"] == "1"
        assert state.calls[0]["price"] > 10
        assert state.calls[0]["pr_type"] != 11

    asyncio.run(run())


def test_helper_cancel_and_partial_fill():
    broker = helper.RemoteBrokerClient.__new__(helper.RemoteBrokerClient)
    broker.account_key = "default"
    broker.sub_account_id = None
    captured = []
    broker._client = SimpleNamespace(request=lambda action, payload: captured.append((action, payload)))
    broker.cancel_order("order-1")
    assert captured[0][1]["idempotency_key"]
    broker.get_orders = lambda: {"1": SimpleNamespace(status="partly_filled"), "2": SimpleNamespace(status="filled")}
    assert set(broker.get_open_orders()) == {"1"}


def test_unknown_exception_preserves_recovery_payload():
    error = helper.SubmissionUnknownError(request(), {"status": "submit_unknown"})
    assert error.idempotency_key == "signal-1"
    assert error.request_payload["amount"] == 100


def test_prepared_request_survives_restart_and_resumes(gateway):
    module, state = gateway
    module._prepare_submission(object(), request())
    assert state.calls == []
    restarted = load_gateway()
    restarted.SUBMISSION_STORE_PATH = module.SUBMISSION_STORE_PATH
    restarted._query_orders = module._query_orders
    restarted._query_positions = module._query_positions
    restarted._place_order_native = module._place_order_native
    assert restarted._place_order(object(), request())["ok"] is True
    assert len(state.calls) == 1


def test_confirmed_result_remains_available_after_broker_archive(gateway):
    module, state = gateway
    module._place_order(object(), request())
    assert module._resolve_submission(request())["order_id"] == "1"
    state.orders.clear()
    result = module._place_order(object(), request())
    assert result["value"]["order_id"] == "1"
    assert not result["value"].get("submit_unknown")
    assert len(state.calls) == 1


def test_post_broker_database_failure_does_not_allow_resubmit(gateway, monkeypatch):
    module, state = gateway
    original = module._submission_db

    class Connection:
        def __init__(self):
            self.connection = original()

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, sql, *args):
            if "SET state='unknown'" in sql:
                raise module.sqlite3.OperationalError("simulated disk failure")
            return self.connection.execute(sql, *args)

    monkeypatch.setattr(module, "_submission_db", Connection)
    first = module._place_order(object(), request())
    assert first["value"]["submit_unknown"] is True
    second = module._place_order(object(), request())
    assert second["value"]["order_id"] == "1"
    assert len(state.calls) == 1


def test_customer_id_cannot_be_reused_for_another_key(gateway):
    module, state = gateway
    module._place_order(object(), request())
    result = module._place_order(object(), request("signal-2", qmt_user_order_id="BT-signal-1"))
    assert result["code"] == "IDEMPOTENCY_CONFLICT"
    assert len(state.calls) == 1


def test_direct_gateway_aliases_require_stable_key(gateway):
    module, state = gateway
    payload = {"stock": "000001.SZ", "volume": 100, "price": 10, "side": "BUY", "account_id": "test-account"}
    assert not module._place_order(object(), payload)["ok"]
    payload["idempotency_key"] = "alias-signal"
    module._place_order(object(), payload)
    module._place_order(object(), payload)
    assert len(state.calls) == 1
    assert state.calls[0]["security"] == "000001.XSHE"


def test_unconfigured_virtual_labels_have_separate_cache_scopes(gateway):
    module, state = gateway

    async def run():
        app, adapter = make_app(module)
        session = SimpleNamespace(account_key="default", sub_account_id=None)
        for label in ("alpha", "beta"):
            payload = request(sub_account_id=label)
            payload.pop("qmt_user_order_id")
            await app._dispatch_broker(session, "place_order", payload)
        assert len(state.calls) == 2
        assert len(app._idempotency_cache) == 2
        assert state.calls[0]["qmt_user_order_id"] != state.calls[1]["qmt_user_order_id"]

    asyncio.run(run())


def test_gateway_preserves_confirmed_result_during_native_return(gateway, monkeypatch):
    module, state = gateway
    original = module._place_order_native

    def native(context, payload):
        response = original(context, payload)
        assert module._resolve_submission(payload)["order_id"] == "1"
        return response

    monkeypatch.setattr(module, "_place_order_native", native)
    module._place_order(object(), request())
    state.orders.clear()
    assert module._resolve_submission(request())["order_id"] == "1"


def test_stale_market_quote_rejected_before_native(gateway):
    module, state = gateway

    async def run():
        app, adapter = make_app(module)

        async def stale(paths, payload):
            return {"ticks": {payload["security"]: {"lastPrice": 10, "time": (time.time() - 3600) * 1000}}}

        adapter.client.post_first = stale
        payload = request(price=None, style={"type": "market"})
        payload.pop("pr_type")
        with pytest.raises(RuntimeError, match="60 秒"):
            await app._dispatch_broker(SimpleNamespace(account_key="default", sub_account_id=None), "place_order", payload)
        assert not state.calls

    asyncio.run(run())


def test_legacy_gateway_is_rejected_before_write(gateway):
    module, state = gateway

    async def run():
        app, adapter = make_app(module)

        async def legacy():
            return {"ready": True}

        adapter.client.health = legacy
        with pytest.raises(RuntimeError, match="升级"):
            await adapter.place_order(adapter.account_router.get("default"), request())
        assert not state.calls

    asyncio.run(run())


def test_hashed_identity_reconciliation(gateway):
    module, state = gateway

    async def run():
        app, adapter = make_app(module)
        payload = request()
        payload.pop("qmt_user_order_id")
        session = SimpleNamespace(account_key="default", sub_account_id=None)
        await app._dispatch_broker(session, "place_order", copy.deepcopy(payload))
        assert state.calls[0]["qmt_user_order_id"] != payload["idempotency_key"]
        restarted, adapter = make_app(module)
        result = await restarted._dispatch_broker(session, "resolve_submission", {
            "write_action": "broker.place_order", "idempotency_key": payload["idempotency_key"], "request_payload": payload,
        })
        assert result["order_id"] == "1"

    asyncio.run(run())


def test_unknown_submission_reserves_risk_budget(gateway, monkeypatch):
    module, state = gateway
    state.crash = True
    with pytest.raises(RuntimeError):
        module._place_order(object(), request())
    state.crash = False
    monkeypatch.setattr(module, "_query_account", lambda *args: {"total_value": 5000})

    async def run():
        app, adapter = make_app(module)
        risk = RiskController()
        risk.config.update(max_position_ratio=50, max_order_value=100000)
        app._risk_by_account["default"] = risk
        with pytest.raises(ValueError, match="仓位"):
            await app._dispatch_broker(SimpleNamespace(account_key="default", sub_account_id=None), "place_order", request("signal-2", amount=200))
        assert len(state.calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["disconnect", "gateway_timeout"])
def test_transport_retries_keep_original_order_identity(monkeypatch, failure):
    client = helper._ShortLivedClient("127.0.0.1", 1, "token", retries=1, retry_interval=0)
    socket = SimpleNamespace(settimeout=lambda value: None, close=lambda: None)
    monkeypatch.setattr(helper.socket, "create_connection", lambda *args, **kwargs: socket)
    sent = []
    current = {}

    def send(sock, message):
        current.clear()
        current.update(message)
        if message["type"] == "request":
            sent.append(copy.deepcopy(message))

    def receive(sock):
        if current["type"] == "handshake":
            return {"type": "handshake_ack", "protocol": 1}
        if len(sent) == 1:
            if failure == "disconnect":
                raise ConnectionResetError("lost response")
            return {"type": "error", "code": "BIG_QMT_GATEWAY_TIMEOUT", "payload": {"broker_called": True}}
        return {"type": "response", "id": current["id"], "payload": {"order_id": "1"}}

    monkeypatch.setattr(client, "_send", send)
    monkeypatch.setattr(client, "_recv", receive)
    assert client.request("broker.place_order", request())["order_id"] == "1"
    assert len(sent) == 2
    assert sent[0]["payload"] == sent[1]["payload"]
