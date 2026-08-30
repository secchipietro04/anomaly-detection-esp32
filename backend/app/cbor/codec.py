# generic cbor encoder and decoder using cbor2 and pydantic v2
from typing import TypeVar, Type, Any
import cbor2
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

def to_cbor(model: BaseModel) -> bytes:
    # serializes pydantic model directly to binary cbor
    return cbor2.dumps(model.model_dump(by_alias=True, exclude_none=True))

def from_cbor(data: bytes, model_cls: Type[T]) -> T:
    # deserializes binary cbor directly to typed pydantic model
    return model_cls.model_validate(cbor2.loads(data))

def dumps_cbor(obj: Any) -> bytes:
    # generic cbor serializer for dicts/primitives
    return cbor2.dumps(obj)

def loads_cbor(data: bytes) -> Any:
    # generic cbor deserializer
    return cbor2.loads(data)

# helpers for compatibility
def encode_runtime_config(config: BaseModel) -> bytes:
    return to_cbor(config)

def decode_runtime_config(data: bytes) -> Any:
    from app.cbor.schemas import RuntimeConfig
    return from_cbor(data, RuntimeConfig)

def encode_cbor(obj: Any) -> bytes:
    return dumps_cbor(obj)

def decode_cbor(data: bytes) -> Any:
    return loads_cbor(data)
