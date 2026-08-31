# unit tests for on-demand request/response handlers
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.cbor.codec import to_cbor, from_cbor
from app.cbor.schemas import (
    RuntimeConfig, EnsembleConfig, RouteEntry,
    AutoencoderModelPackage, RouterModelPackage, MemoryModelPackage,
    ModelType, LossMode, ArchitectureTag, InferencePacket, NodeHealthInfo
)
from app.database.models import ModelPackageModel, EnsembleConfigModel, NodeModel
from app.mqtt.handlers import (
    handle_model_fetch_request,
    handle_ensemble_fetch_request,
    handle_config_fetch_request,
    handle_inference_packet,
    handle_node_health
)

@pytest.mark.asyncio
async def test_handle_model_fetch_request():
    session = AsyncMock()
    publisher = AsyncMock()

    # mock model record in DB
    mock_model = ModelPackageModel(
        id=2001,
        node_id="node_abc",
        tag=int(ArchitectureTag.DA),
        model_type=int(ModelType.AUTOENCODER),
        tflite_binary=b"MOCK_TFLITE_FLATBUFFER",
        config={
            "accel_bins": 128,
            "gyro_bins": 128,
            "tsteps": 16,
            "limit": 0.15,
            "loss": int(LossMode.LOG_MSE),
            "skip": 2
        }
    )
    session.get.return_value = mock_model

    await handle_model_fetch_request(
        node_id="node_abc",
        model_type="submodel",
        model_id="2001",
        session=session,
        publisher=publisher
    )

    publisher.publish_model.assert_called_once()
    call_args = publisher.publish_model.call_args[1]
    assert call_args["node_id"] == "node_abc"
    assert call_args["model_type"] == "submodel"
    assert call_args["model_id"] == 2001
    assert call_args["retain"] is False

    # decode payload and verify it's a valid AutoencoderModelPackage
    decoded_pkg = from_cbor(call_args["payload"], AutoencoderModelPackage)
    assert decoded_pkg.m_id == 2001
    assert decoded_pkg.data == b"MOCK_TFLITE_FLATBUFFER"
    assert decoded_pkg.accel_bins == 128
    assert decoded_pkg.limit == 0.15

@pytest.mark.asyncio
async def test_handle_ensemble_fetch_request():
    session = AsyncMock()
    publisher = AsyncMock()

    mock_res = MagicMock()
    mock_ens = EnsembleConfigModel(
        id=123456,
        node_id="node_abc",
        router_model_id=1001,
        memory_model_id=None,
        warmup=10,
        routes=[{"out_ix": 0, "m_id": 2001}, {"out_ix": 1, "m_id": 2002}],
        total_size_bytes=50000
    )
    mock_res.scalar_one_or_none.return_value = mock_ens
    session.execute.return_value = mock_res

    await handle_ensemble_fetch_request(
        node_id="node_abc",
        payload=b"",
        session=session,
        publisher=publisher
    )

    publisher.publish_ensemble.assert_called_once()
    call_args = publisher.publish_ensemble.call_args[0]
    assert call_args[0] == "node_abc"
    decoded_ens = from_cbor(call_args[1], EnsembleConfig)
    assert decoded_ens.r_m_id == 1001
    assert len(decoded_ens.routes) == 2
    assert decoded_ens.routes[0].m_id == 2001

@pytest.mark.asyncio
async def test_handle_config_fetch_request():
    session = AsyncMock()
    publisher = AsyncMock()

    mock_node = NodeModel(
        node_id="node_abc",
        current_config={"rate": 3840.0, "mode": 1, "batch": 512, "beat": 10, "sd_en": True}
    )
    session.get.return_value = mock_node

    await handle_config_fetch_request(
        node_id="node_abc",
        payload=b"",
        session=session,
        publisher=publisher
    )

    publisher.publish_config.assert_called_once()
    call_args = publisher.publish_config.call_args[0]
    assert call_args[0] == "node_abc"
    decoded_cfg = from_cbor(call_args[1], RuntimeConfig)
    assert decoded_cfg.rate == 3840.0
    assert decoded_cfg.mode == 1
