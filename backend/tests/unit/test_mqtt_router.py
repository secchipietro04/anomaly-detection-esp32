# unit test for fastapi-style mqtt router
from app.mqtt.router import MQTTRouter

def test_mqtt_router_exact_and_param_matching():
    router = MQTTRouter(prefix="v1")

    @router.subscribe("v1/{node_id}/data/sensor")
    def on_sensor(node_id, payload):
        return f"sensor-{node_id}"

    @router.subscribe("v1/{node_id}/models/{model_type}/{model_id}")
    def on_model(node_id, model_type, model_id, payload):
        return f"{node_id}-{model_type}-{model_id}"

    # match sensor route
    handler, args = router.resolve("v1/node_123/data/sensor")
    assert handler == on_sensor
    assert args == {"node_id": "node_123"}

    # match model route
    handler, args = router.resolve("v1/node_456/models/router/1001")
    assert handler == on_model
    assert args == {"node_id": "node_456", "model_type": "router", "model_id": "1001"}

    # no match
    res = router.resolve("v1/node_123/unknown/topic")
    assert res is None

def test_subscription_topics_generation():
    router = MQTTRouter(prefix="v1")
    router.subscribe("v1/{node_id}/data/sensor")(lambda: None)
    router.subscribe("v1/{node_id}/info/caps")(lambda: None)
    topics = router.get_subscription_topics()
    assert "v1/+/data/sensor" in topics
    assert "v1/+/info/caps" in topics
