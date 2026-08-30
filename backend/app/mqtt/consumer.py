# backward compatibility wrapper for mqtt consumer
from app.mqtt.client import MQTTClientManager
from app.mqtt.router import MQTTRouter
from app.mqtt.handlers import router as default_router

class MQTTConsumer(MQTTClientManager):
    # alias to client manager
    pass
