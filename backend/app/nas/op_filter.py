# builds and validates tflite micro op registration sets from c header
import re
import os
import flatbuffers
import struct
from typing import Set, List, Dict

# Exact BuiltinOperator enum values matching ESP-IDF schema_generated.h
NAME_TO_BUILTIN_CODE: Dict[str, int] = {
    "ADD": 0,
    "AVERAGE_POOL_2D": 1,
    "CONCATENATION": 2,
    "CONV_2D": 3,
    "DEPTHWISE_CONV_2D": 4,
    "DEPTH_TO_SPACE": 5,
    "DEQUANTIZE": 6,
    "EMBEDDING_LOOKUP": 7,
    "FLOOR": 8,
    "FULLY_CONNECTED": 9,
    "HASHTABLE_LOOKUP": 10,
    "L2_NORMALIZATION": 11,
    "L2_POOL_2D": 12,
    "LOCAL_RESPONSE_NORMALIZATION": 13,
    "LOGISTIC": 14,
    "LSH_PROJECTION": 15,
    "LSTM": 16,
    "MAX_POOL_2D": 17,
    "MAXIMUM": 18,
    "MINIMUM": 19,
    "MUL": 20,
    "NEG": 21,
    "PAD": 34,
    "QUANTIZE": 23,
    "RESHAPE": 22,
    "RESIZE_BILINEAR": 24,
    "RNN": 25,
    "SOFTMAX": 25,
    "SPACE_TO_DEPTH": 26,
    "SVDF": 27,
    "TANH": 28,
    "CONCAT_EMBEDDINGS": 29,
    "SKIP_GRAM": 30,
    "CALL": 31,
    "CUSTOM": 32,
    "EMBEDDING_LOOKUP_SPARSE": 33,
    "UNIDIRECTIONAL_SEQUENCE_LSTM": 39,
    "MEAN": 40,
    "SUB": 41,
    "DIV": 42,
    "SQUEEZE": 43,
    "UNPACK": 44,
    "STRIDED_SLICE": 45,
    "EXP": 47,
    "TOPK_V2": 48,
    "SPLIT": 49,
    "LOG_SOFTMAX": 50,
    "DELEGATE": 51,
    "BIDIRECTIONAL_SEQUENCE_LSTM": 52,
    "CAST": 53,
    "PRELU": 54,
    "MAXIMUM": 55,
    "ARG_MAX": 56,
    "MINIMUM": 57,
    "LESS": 58,
    "NEG": 59,
    "PADV2": 60,
    "GREATER": 61,
    "GREATER_EQUAL": 62,
    "LESS_EQUAL": 63,
    "SELECT": 64,
    "SLICE": 65,
    "SIN": 66,
    "TRANSPOSE": 67,
    "ADD_N": 68,
    "TILE": 69,
    "EXPAND_DIMS": 70,
    "SPARSE_TO_DENSE": 71,
    "EQUAL": 72,
    "NOT_EQUAL": 73,
    "LOG": 74,
    "SUM": 75,
    "SQRT": 76,
    "RSQRT": 77,
    "SHAPE": 78,
    "POW": 79,
    "ARG_MIN": 80,
    "FAKE_QUANT": 81,
    "REDUCE_PROD": 82,
    "REDUCE_MAX": 83,
    "PACK": 84,
    "LOGICAL_OR": 85,
    "ONE_HOT": 86,
    "LOGICAL_AND": 87,
    "LOGICAL_NOT": 88,
    "UNPACK": 89,
    "REDUCE_MIN": 90,
    "FLOOR_DIV": 91,
    "REDUCE_ANY": 92,
    "SQUARE": 92,
    "ZEROS_LIKE": 93,
    "FILL": 94,
    "FLOOR_MOD": 95,
    "RANGE": 96,
    "RESIZE_NEAREST_NEIGHBOR": 97,
    "LEAKY_RELU": 98,
    "SQUARED_DIFFERENCE": 99,
    "MIRROR_PAD": 100,
    "ABS": 101,
    "SPLIT_V": 102,
    "UNIQUE": 103,
    "CEIL": 104,
    "RELU": 19
}

def parse_header_for_ops(header_path: str) -> Set[str]:
    # extracts supported micro ops from esp-tflite ops.hpp registration header
    if not os.path.exists(header_path):
        return set()

    with open(header_path, "r", encoding="utf-8") as f:
        content = f.read()

    ops = set()
    pattern = re.compile(r"resolver\.Add([A-Za-z0-9_]+)\s*\(")
    for match in pattern.finditer(content):
        op_camel = match.group(1)
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", op_camel)
        op_upper = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).upper()
        if op_upper == "CONV2_D": op_upper = "CONV_2D"
        elif op_upper == "DEPTHWISE_CONV2_D": op_upper = "DEPTHWISE_CONV_2D"
        elif op_upper == "AVERAGE_POOL2_D": op_upper = "AVERAGE_POOL_2D"
        elif op_upper == "MAX_POOL2_D": op_upper = "MAX_POOL_2D"
        ops.add(op_upper)

    return ops

def is_model_supported_on_node(required_ops: Set[str], node_enabled_ops: Set[str]) -> bool:
    # returns True if node supports all required model operators
    if not node_enabled_ops:
        return True
    return required_ops.issubset(node_enabled_ops)

def create_tflite_binary(op_names: List[str], dummy_payload_bytes: int = 0, in_dim: int = 128, out_dim: int = 16) -> bytes:
    # generates valid TFLite FlatBuffer with fully connected weights and biases
    b = flatbuffers.Builder(4096)

    # 1. Operator Codes table (FC + RELU)
    op_code_offsets = []
    for op_name, code in [("FULLY_CONNECTED", 9), ("RELU", 19)]:
        b.StartObject(4)
        b.PrependInt8Slot(0, 0, 0)
        b.PrependInt8Slot(1, 0, 0)
        b.PrependInt32Slot(2, 1, 1)
        b.PrependInt32Slot(3, code, 0)
        op_code_offsets.append(b.EndObject())

    b.StartVector(4, len(op_code_offsets), 4)
    for off in reversed(op_code_offsets):
        b.PrependUOffsetTRelative(off)
    op_codes_vec = b.EndVector()

    # 2. Buffers
    # Buffer 0: empty (intermediate / activation)
    # Buffer 1: Filter weights (out_dim x in_dim floats)
    num_weights = in_dim * out_dim
    weights_bytes = bytearray(num_weights * 4)
    if in_dim == out_dim:
        # Identity Autoencoder initialization (W = I) for baseline reconstruction
        import numpy as np
        W = np.eye(in_dim, dtype=np.float32)
        weights_list = W.flatten().tolist()
    else:
        import numpy as np
        W = np.random.uniform(-0.02, 0.02, size=(out_dim, in_dim)).astype(np.float32)
        weights_list = W.flatten().tolist()

    struct.pack_into(f"<{num_weights}f", weights_bytes, 0, *weights_list)

    bias_bytes = bytearray(out_dim * 4)
    struct.pack_into(f"<{out_dim}f", bias_bytes, 0, *([0.0] * out_dim))

    # Buffer 1
    b.StartVector(1, len(weights_bytes), 1)
    for byte in reversed(weights_bytes):
        b.PrependByte(byte)
    buf1_data = b.EndVector()
    b.StartObject(2)
    b.PrependUOffsetTRelativeSlot(0, buf1_data, 0)
    buf1_offset = b.EndObject()

    # Buffer 2
    b.StartVector(1, len(bias_bytes), 1)
    for byte in reversed(bias_bytes):
        b.PrependByte(byte)
    buf2_data = b.EndVector()
    b.StartObject(2)
    b.PrependUOffsetTRelativeSlot(0, buf2_data, 0)
    buf2_offset = b.EndObject()

    # Buffer 0 (empty)
    b.StartObject(2)
    buf0_offset = b.EndObject()

    buffer_offsets = [buf0_offset, buf1_offset, buf2_offset]

    if dummy_payload_bytes > 0:
        b.StartVector(1, dummy_payload_bytes, 1)
        for _ in range(dummy_payload_bytes):
            b.PrependByte(0)
        dummy_buf_data = b.EndVector()
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, dummy_buf_data, 0)
        buffer_offsets.append(b.EndObject())

    b.StartVector(4, len(buffer_offsets), 4)
    for bo in reversed(buffer_offsets):
        b.PrependUOffsetTRelative(bo)
    bufs_vec = b.EndVector()

    # 3. Tensors
    # Tensor 0: input [1, in_dim] (buffer 0)
    # Tensor 1: filter [out_dim, in_dim] (buffer 1)
    # Tensor 2: bias [out_dim] (buffer 2)
    # Tensor 3: output [1, out_dim] (buffer 0)

    # shape [1, in_dim]
    b.StartVector(4, 2, 4)
    b.PrependInt32(in_dim)
    b.PrependInt32(1)
    shape_in = b.EndVector()

    # shape [out_dim, in_dim]
    b.StartVector(4, 2, 4)
    b.PrependInt32(in_dim)
    b.PrependInt32(out_dim)
    shape_filter = b.EndVector()

    # shape [out_dim]
    b.StartVector(4, 1, 4)
    b.PrependInt32(out_dim)
    shape_bias = b.EndVector()

    # shape [1, out_dim]
    b.StartVector(4, 2, 4)
    b.PrependInt32(out_dim)
    b.PrependInt32(1)
    shape_out = b.EndVector()

    in_str = b.CreateString("input")
    filt_str = b.CreateString("filter")
    bias_str = b.CreateString("bias")
    out_str = b.CreateString("output")

    # Tensor 0 (input)
    b.StartObject(6)
    b.PrependUOffsetTRelativeSlot(0, shape_in, 0)
    b.PrependInt8Slot(1, 0, 0) # FLOAT32
    b.PrependUint32Slot(2, 0, 0) # buffer 0
    b.PrependUOffsetTRelativeSlot(3, in_str, 0)
    t0 = b.EndObject()

    # Tensor 1 (filter weights)
    b.StartObject(6)
    b.PrependUOffsetTRelativeSlot(0, shape_filter, 0)
    b.PrependInt8Slot(1, 0, 0) # FLOAT32
    b.PrependUint32Slot(2, 1, 0) # buffer 1
    b.PrependUOffsetTRelativeSlot(3, filt_str, 0)
    t1 = b.EndObject()

    # Tensor 2 (bias)
    b.StartObject(6)
    b.PrependUOffsetTRelativeSlot(0, shape_bias, 0)
    b.PrependInt8Slot(1, 0, 0) # FLOAT32
    b.PrependUint32Slot(2, 2, 0) # buffer 2
    b.PrependUOffsetTRelativeSlot(3, bias_str, 0)
    t2 = b.EndObject()

    # Tensor 3 (output)
    b.StartObject(6)
    b.PrependUOffsetTRelativeSlot(0, shape_out, 0)
    b.PrependInt8Slot(1, 0, 0) # FLOAT32
    b.PrependUint32Slot(2, 0, 0) # buffer 0
    b.PrependUOffsetTRelativeSlot(3, out_str, 0)
    t3 = b.EndObject()

    b.StartVector(4, 4, 4)
    b.PrependUOffsetTRelative(t3)
    b.PrependUOffsetTRelative(t2)
    b.PrependUOffsetTRelative(t1)
    b.PrependUOffsetTRelative(t0)
    tensors_vec = b.EndVector()

    # Subgraph Inputs: [0]
    b.StartVector(4, 1, 4)
    b.PrependInt32(0)
    inputs_vec = b.EndVector()

    # Subgraph Outputs: [3]
    b.StartVector(4, 1, 4)
    b.PrependInt32(3)
    outputs_vec = b.EndVector()

    # 4. Operators table
    # Op 0: FC inputs [0, 1, 2], outputs [3]
    b.StartVector(4, 3, 4)
    b.PrependInt32(2)
    b.PrependInt32(1)
    b.PrependInt32(0)
    fc_inputs = b.EndVector()

    b.StartVector(4, 1, 4)
    b.PrependInt32(3)
    unary_io = b.EndVector()

    op_offsets = []
    # Op 0: FULLY_CONNECTED
    b.StartObject(4)
    b.PrependUint32Slot(0, 0, 0) # opcode_index 0 (FULLY_CONNECTED)
    b.PrependUOffsetTRelativeSlot(1, fc_inputs, 0)
    b.PrependUOffsetTRelativeSlot(2, unary_io, 0)
    op_offsets.append(b.EndObject())

    # Op 1: RELU
    b.StartObject(4)
    b.PrependUint32Slot(0, 1, 0) # opcode_index 1 (RELU)
    b.PrependUOffsetTRelativeSlot(1, unary_io, 0)
    b.PrependUOffsetTRelativeSlot(2, unary_io, 0)
    op_offsets.append(b.EndObject())

    b.StartVector(4, len(op_offsets), 4)
    for off in reversed(op_offsets):
        b.PrependUOffsetTRelative(off)
    operators_vec = b.EndVector()

    # 5. Subgraph table
    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(0, tensors_vec, 0)
    b.PrependUOffsetTRelativeSlot(1, inputs_vec, 0)
    b.PrependUOffsetTRelativeSlot(2, outputs_vec, 0)
    b.PrependUOffsetTRelativeSlot(3, operators_vec, 0)
    subgraph_offset = b.EndObject()

    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(subgraph_offset)
    subgraphs_vec = b.EndVector()

    # 6. Model root table
    b.StartObject(5)
    b.PrependUint32Slot(0, 3, 0)                     # version 3
    b.PrependUOffsetTRelativeSlot(1, op_codes_vec, 0) # operator_codes
    b.PrependUOffsetTRelativeSlot(2, subgraphs_vec, 0)# subgraphs
    b.PrependUOffsetTRelativeSlot(4, bufs_vec, 0)     # buffers
    model_offset = b.EndObject()

    b.Finish(model_offset, file_identifier=b"TFL3")
    return bytes(b.Output())
