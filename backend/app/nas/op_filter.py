# tflite operator extraction and enabled_ops filtering
import struct
from typing import List, Set, Tuple, Union, Optional
import flatbuffers

# 118 operators compiled in firmware tflite micro wrapper
ALL_SUPPORTED_OPS: Set[str] = {
    "ABS", "ADD", "ADD_N", "ARG_MAX", "ARG_MIN", "ASSIGN_VARIABLE",
    "AVERAGE_POOL_2D", "BATCH_MATMUL", "BATCH_TO_SPACE_ND", "BROADCAST_ARGS",
    "BROADCAST_TO", "CALL_ONCE", "CAST", "CEIL", "CIRCULAR_BUFFER",
    "CONCATENATION", "CONV_2D", "COS", "CUMSUM", "DECODE",
    "DELAY", "DEPTH_TO_SPACE", "DEPTHWISE_CONV_2D", "DEQUANTIZE",
    "DETECTION_POSTPROCESS", "DIV", "DYNAMIC_UPDATE_SLICE", "EMBEDDING_LOOKUP",
    "ENERGY", "ELU", "EQUAL", "ETHOSU", "EXP", "EXPAND_DIMS",
    "FFT_AUTO_SCALE", "FILL", "FILTER_BANK", "FILTER_BANK_LOG",
    "FILTER_BANK_SQUARE_ROOT", "FILTER_BANK_SPECTRAL_SUBTRACTION", "FLOOR",
    "FLOOR_DIV", "FLOOR_MOD", "FRAMER", "FULLY_CONNECTED", "GATHER",
    "GATHER_ND", "GREATER", "GREATER_EQUAL", "HARD_SWISH", "IF",
    "IRFFT", "L2_NORMALIZATION", "L2_POOL_2D", "LEAKY_RELU", "LESS",
    "LESS_EQUAL", "LOG", "LOGICAL_AND", "LOGICAL_NOT", "LOGICAL_OR",
    "LOGISTIC", "LOG_SOFTMAX", "MAXIMUM", "MAX_POOL_2D", "MIRROR_PAD",
    "MEAN", "MINIMUM", "MUL", "NEG", "NOT_EQUAL", "OVERLAP_ADD",
    "PACK", "PAD", "PAD_V2", "PCAN", "PRELU", "QUANTIZE",
    "READ_VARIABLE", "REDUCE_ALL", "REDUCE_MAX", "REDUCE_MIN", "RELU",
    "RELU6", "RESHAPE", "RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR",
    "REVERSE_V2", "RFFT", "ROUND", "RSQRT", "SELECT_V2", "SHAPE",
    "SIN", "SLICE", "SOFTMAX", "SPACE_TO_BATCH_ND", "SPACE_TO_DEPTH",
    "SPLIT", "SPLIT_V", "SQUEEZE", "SQRT", "SQUARE", "SQUARED_DIFFERENCE",
    "STRIDED_SLICE", "STACKER", "SUB", "SUM", "SVDF", "TANH",
    "TRANSPOSE_CONV", "TRANSPOSE", "UNPACK", "UNIDIRECTIONAL_SEQUENCE_LSTM",
    "VAR_HANDLE", "WHILE", "WINDOW", "ZEROS_LIKE"
}

# TFLite standard builtin operator opcode mapping
BUILTIN_CODE_TO_NAME = {
    0: "ADD", 1: "AVERAGE_POOL_2D", 2: "CONCATENATION", 3: "CONV_2D",
    4: "DEPTHWISE_CONV_2D", 5: "DEPTH_TO_SPACE", 6: "DEQUANTIZE",
    7: "EMBEDDING_LOOKUP", 8: "FLOOR", 9: "FULLY_CONNECTED",
    10: "HASHTABLE_LOOKUP", 11: "L2_NORMALIZATION", 12: "L2_POOL_2D",
    13: "LOCAL_RESPONSE_NORMALIZATION", 14: "LOGISTIC", 15: "LOG_SOFTMAX",
    16: "LSTM", 17: "MAX_POOL_2D", 18: "MAXIMUM", 19: "MINIMUM",
    20: "MUL", 21: "NEG", 22: "PAD", 23: "QUANTIZE", 24: "RESHAPE",
    25: "RESIZE_BILINEAR", 26: "RNN", 27: "SOFTMAX", 28: "SPACE_TO_DEPTH",
    29: "SVDF", 30: "TANH", 31: "CONCAT_EMBEDDINGS", 32: "SKIP_GRAM",
    33: "CALL", 34: "CUSTOM", 35: "EMBEDDING_LOOKUP_SPARSE", 36: "PAD_V2",
    37: "SPLIT", 38: "SQRT", 39: "SQUARED_DIFFERENCE", 40: "FAKE_QUANT",
    41: "TRANSPOSE_CONV", 42: "EXP", 43: "TOPK_V2", 44: "SPLIT_V",
    45: "CAST", 46: "ARG_MAX", 47: "DEALLOCATE_BUFFER", 48: "ARG_MIN",
    49: "EXPAND_DIMS", 50: "LOGICAL_OR", 51: "ONE_HOT", 52: "LOGICAL_AND",
    53: "LOGICAL_NOT", 54: "LESS", 55: "REDUCE_MAX", 56: "GREATER",
    57: "GREATER_EQUAL", 58: "REDUCE_MIN", 59: "RANK", 60: "SQUARE",
    61: "SHAPE", 62: "EQUAL", 63: "NOT_EQUAL", 64: "MEAN",
    65: "REDUCE_ALL", 66: "SQUEEZE", 67: "FILL", 68: "CEIL",
    69: "ELU", 70: "COS", 71: "HARD_SWISH", 72: "LEAKY_RELU",
    73: "MATRIX_DIAG", 74: "REDUCE_PROD", 75: "SELECT", 76: "SLICE",
    77: "SIN", 78: "TRANSPOSE", 79: "STRIDED_SLICE",
    80: "UNIDIRECTIONAL_SEQUENCE_LSTM", 81: "MATRIX_SET_DIAG",
    82: "RESIZE_NEAREST_NEIGHBOR", 83: "ROUND", 84: "RSQRT",
    85: "UNIDIRECTIONAL_SEQUENCE_RNN", 86: "FLOOR_DIV", 87: "SUB",
    88: "ZEROS_LIKE", 89: "PRELU", 90: "MAXIMUM_V2", 91: "MINIMUM_V2",
    92: "CUMSUM", 93: "CALL_ONCE", 94: "BROADCAST_TO", 95: "RFFT2D",
    96: "CONV_3D", 97: "IMAG", 98: "REAL", 99: "COMPLEX_ABS",
    100: "PACK", 101: "UNPACK", 102: "FLOOR_MOD", 103: "BATCH_TO_SPACE_ND",
    104: "SPACE_TO_BATCH_ND", 105: "REVERSE_V2", 106: "ABS", 107: "WHILE",
    108: "DELEGATE", 109: "READ_VARIABLE", 110: "ASSIGN_VARIABLE",
    111: "VAR_HANDLE", 112: "BATCH_MATMUL", 113: "DYNAMIC_UPDATE_SLICE",
    114: "BROADCAST_ARGS", 115: "ETHOSU", 116: "CIRCULAR_BUFFER",
    117: "DECODE", 118: "DELAY", 119: "ENERGY", 120: "FFT_AUTO_SCALE",
    121: "FILTER_BANK", 122: "FILTER_BANK_LOG", 123: "FILTER_BANK_SQUARE_ROOT",
    124: "FILTER_BANK_SPECTRAL_SUBTRACTION", 125: "FRAMER", 126: "IF",
    127: "IRFFT", 128: "LOG", 129: "MIRROR_PAD", 130: "OVERLAP_ADD",
    131: "PCAN", 132: "RELU", 133: "RELU6", 134: "RFFT",
    135: "SELECT_V2", 136: "STACKER", 137: "SUM", 138: "WINDOW",
    139: "ADD_N", 140: "LESS_EQUAL"
}

NAME_TO_BUILTIN_CODE = {name: code for code, name in BUILTIN_CODE_TO_NAME.items()}

def extract_tflite_ops(tflite_data: bytes) -> List[str]:
    # parse flatbuffer binary to extract operator names
    if not isinstance(tflite_data, (bytes, bytearray, memoryview)) or len(tflite_data) < 8:
        return []
    
    data = bytes(tflite_data)
    try:
        root_offset = struct.unpack_from("<I", data, 0)[0]
        model_pos = root_offset
        if model_pos >= len(data):
            return []
        
        vtable_offset = struct.unpack_from("<i", data, model_pos)[0]
        vtable_pos = model_pos - vtable_offset
        if vtable_pos < 0 or vtable_pos + 2 > len(data):
            return []
        
        vtable_len = struct.unpack_from("<H", data, vtable_pos)[0]
        
        # operator_codes is field 1 in Model table -> offset at vtable_pos + 4 + 2*1 = vtable_pos + 6
        if vtable_len <= 6:
            return []
        
        op_codes_field_offset = struct.unpack_from("<H", data, vtable_pos + 6)[0]
        if op_codes_field_offset == 0:
            return []
        
        op_codes_vec_pos = model_pos + op_codes_field_offset
        if op_codes_vec_pos + 4 > len(data):
            return []
        
        vec_start = op_codes_vec_pos + struct.unpack_from("<I", data, op_codes_vec_pos)[0]
        if vec_start + 4 > len(data):
            return []
            
        vec_len = struct.unpack_from("<I", data, vec_start)[0]
        
        ops: List[str] = []
        for i in range(vec_len):
            elem_ptr = vec_start + 4 + 4 * i
            if elem_ptr + 4 > len(data):
                continue
            op_code_pos = elem_ptr + struct.unpack_from("<I", data, elem_ptr)[0]
            if op_code_pos >= len(data):
                continue
                
            oc_vtable_offset = struct.unpack_from("<i", data, op_code_pos)[0]
            oc_vtable_pos = op_code_pos - oc_vtable_offset
            if oc_vtable_pos < 0 or oc_vtable_pos + 2 > len(data):
                continue
            oc_vtable_len = struct.unpack_from("<H", data, oc_vtable_pos)[0]
            
            builtin_code = 0
            # field 3: builtin_code (int32) at vtable_pos + 10
            if oc_vtable_len > 10:
                b_off = struct.unpack_from("<H", data, oc_vtable_pos + 10)[0]
                if b_off != 0 and op_code_pos + b_off + 4 <= len(data):
                    builtin_code = struct.unpack_from("<i", data, op_code_pos + b_off)[0]
            
            # fallback field 0: deprecated_builtin_code (int8) at vtable_pos + 4
            if builtin_code == 0 and oc_vtable_len > 4:
                d_off = struct.unpack_from("<H", data, oc_vtable_pos + 4)[0]
                if d_off != 0 and op_code_pos + d_off + 1 <= len(data):
                    builtin_code = struct.unpack_from("<b", data, op_code_pos + d_off)[0]
                    
            # custom code string if present (field 1 at vtable_pos + 6)
            custom_name = None
            if oc_vtable_len > 6:
                c_off = struct.unpack_from("<H", data, oc_vtable_pos + 6)[0]
                if c_off != 0 and op_code_pos + c_off + 4 <= len(data):
                    c_str_pos = op_code_pos + c_off + struct.unpack_from("<I", data, op_code_pos + c_off)[0]
                    if c_str_pos + 4 <= len(data):
                        c_len = struct.unpack_from("<I", data, c_str_pos)[0]
                        if c_str_pos + 4 + c_len <= len(data):
                            custom_name = data[c_str_pos + 4: c_str_pos + 4 + c_len].decode("utf-8", errors="ignore")
                            
            op_name = custom_name if custom_name else BUILTIN_CODE_TO_NAME.get(builtin_code, f"OP_{builtin_code}")
            if op_name not in ops:
                ops.append(op_name)
        return ops
    except Exception:
        return []

def validate_candidate_architecture(
    candidate_ops: List[str],
    enabled_ops: List[str]
) -> Tuple[bool, List[str]]:
    # check candidate operator requirements against node enabled_ops
    enabled_set = set(enabled_ops)
    missing = [op for op in candidate_ops if op not in enabled_set]
    is_valid = len(missing) == 0
    return is_valid, missing

def is_model_compatible(
    model: Union[bytes, List[str]],
    enabled_ops: List[str]
) -> bool:
    # return True if model only uses enabled ops
    if isinstance(model, (bytes, bytearray, memoryview)):
        ops = extract_tflite_ops(bytes(model))
    else:
        ops = list(model)
    is_valid, _ = validate_candidate_architecture(ops, enabled_ops)
    return is_valid

def create_tflite_binary(
    op_names: List[str],
    dummy_payload_bytes: int = 0
) -> bytes:
    # generate a valid tflite flatbuffer containing the specified operators
    b = flatbuffers.Builder(1024 + dummy_payload_bytes)
    
    op_code_offsets = []
    for name in op_names:
        code = NAME_TO_BUILTIN_CODE.get(name, 34)  # 34 is CUSTOM
        custom_str_offset = None
        if code == 34 or name not in NAME_TO_BUILTIN_CODE:
            custom_str_offset = b.CreateString(name)
            
        b.StartObject(4)
        if code < 127:
            b.PrependInt8Slot(0, code, 0)
        if custom_str_offset is not None:
            b.PrependUOffsetTRelativeSlot(1, custom_str_offset, 0)
        b.PrependInt32Slot(2, 1, 1)
        b.PrependInt32Slot(3, code, 0)
        op_code_offsets.append(b.EndObject())
        
    b.StartVector(4, len(op_code_offsets), 4)
    for off in reversed(op_code_offsets):
        b.PrependUOffsetTRelative(off)
    op_codes_vec = b.EndVector()
    
    # operators inside subgraph
    op_offsets = []
    for i in range(len(op_code_offsets)):
        b.StartObject(4)
        b.PrependUint32Slot(0, i, 0)
        op_offsets.append(b.EndObject())
        
    b.StartVector(4, len(op_offsets), 4)
    for off in reversed(op_offsets):
        b.PrependUOffsetTRelative(off)
    operators_vec = b.EndVector()
    
    # dummy buffers if requested for size simulation
    buffer_offsets = []
    if dummy_payload_bytes > 0:
        b.StartVector(1, dummy_payload_bytes, 1)
        # write dummy bytes
        for _ in range(dummy_payload_bytes):
            b.PrependByte(0)
        buf_data = b.EndVector()
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, buf_data, 0)
        buffer_offsets.append(b.EndObject())
        
    # subgraph table
    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(3, operators_vec, 0)
    subgraph_offset = b.EndObject()
    
    b.StartVector(4, 1, 4)
    b.PrependUOffsetTRelative(subgraph_offset)
    subgraphs_vec = b.EndVector()
    
    # model root table
    b.StartObject(5)
    b.PrependUint32Slot(0, 3, 0)  # version 3
    b.PrependUOffsetTRelativeSlot(1, op_codes_vec, 0)
    b.PrependUOffsetTRelativeSlot(2, subgraphs_vec, 0)
    if buffer_offsets:
        b.StartVector(4, len(buffer_offsets), 4)
        for bo in reversed(buffer_offsets):
            b.PrependUOffsetTRelative(bo)
        bufs_vec = b.EndVector()
        b.PrependUOffsetTRelativeSlot(4, bufs_vec, 0)
    model_offset = b.EndObject()
    
    b.Finish(model_offset, file_identifier=b"TFL3")
    return bytes(b.Output())
