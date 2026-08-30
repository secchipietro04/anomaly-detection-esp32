# unit test for generic cbor codec and schemas
import pytest
from app.cbor.schemas import (
    RuntimeConfig, StreamMode, NodeCapabilities, NodeHealthInfo,
    InferencePacket, Segment, EnsembleConfig, RouteEntry
)
from app.cbor.codec import to_cbor, from_cbor, dumps_cbor, loads_cbor

def test_runtime_config_generic_codec():
    cfg = RuntimeConfig(mode=int(StreamMode.CONTINUOUS), rate=800.0, beat=30, batch=4, sd_en=False)
    encoded = to_cbor(cfg)
    assert isinstance(encoded, bytes)
    decoded = from_cbor(encoded, RuntimeConfig)
    assert decoded.mode == 1
    assert abs(decoded.rate - 800.0) < 1e-3
    assert decoded.beat == 30
    assert decoded.batch == 4

def test_ensemble_config_generic_codec():
    ens = EnsembleConfig(
        warmup=10,
        r_m_id=101,
        mem_id=201,
        routes=[RouteEntry(out_ix=0, m_id=301), RouteEntry(out_ix=1, m_id=302)]
    )
    encoded = to_cbor(ens)
    decoded = from_cbor(encoded, EnsembleConfig)
    assert decoded.warmup == 10
    assert decoded.r_m_id == 101
    assert len(decoded.routes) == 2
    assert decoded.routes[1].m_id == 302

def test_generic_primitives():
    data = {"node_id": "esp32-01", "val": [1.5, 2.5]}
    encoded = dumps_cbor(data)
    decoded = loads_cbor(encoded)
    assert decoded["node_id"] == "esp32-01"
    assert decoded["val"] == [1.5, 2.5]
