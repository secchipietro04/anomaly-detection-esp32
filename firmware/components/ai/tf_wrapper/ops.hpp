#ifndef TFLITE_OPS_HPP
#define TFLITE_OPS_HPP

#include <stddef.h>

#ifdef __cplusplus

/* Build-time operator counting */
constexpr unsigned int kMaxOpCount =
#ifdef CONFIG_TFL_USE_ABS
    1 +
#endif
#ifdef CONFIG_TFL_USE_ADD
    1 +
#endif
#ifdef CONFIG_TFL_USE_ADD_N
    1 +
#endif
#ifdef CONFIG_TFL_USE_ARG_MAX
    1 +
#endif
#ifdef CONFIG_TFL_USE_ARG_MIN
    1 +
#endif
#ifdef CONFIG_TFL_USE_ASSIGN_VARIABLE
    1 +
#endif
#ifdef CONFIG_TFL_USE_AVERAGE_POOL_2D
    1 +
#endif
#ifdef CONFIG_TFL_USE_BATCH_MATMUL
    1 +
#endif
#ifdef CONFIG_TFL_USE_BATCH_TO_SPACE_ND
    1 +
#endif
#ifdef CONFIG_TFL_USE_BROADCAST_ARGS
    1 +
#endif
#ifdef CONFIG_TFL_USE_BROADCAST_TO
    1 +
#endif
#ifdef CONFIG_TFL_USE_CALL_ONCE
    1 +
#endif
#ifdef CONFIG_TFL_USE_CAST
    1 +
#endif
#ifdef CONFIG_TFL_USE_CEIL
    1 +
#endif
#ifdef CONFIG_TFL_USE_CIRCULAR_BUFFER
    1 +
#endif
#ifdef CONFIG_TFL_USE_CONCATENATION
    1 +
#endif
#ifdef CONFIG_TFL_USE_CONV_2D
    1 +
#endif
#ifdef CONFIG_TFL_USE_COS
    1 +
#endif
#ifdef CONFIG_TFL_USE_CUMSUM
    1 +
#endif
#ifdef CONFIG_TFL_USE_DECODE
    1 +
#endif
#ifdef CONFIG_TFL_USE_DELAY
    1 +
#endif
#ifdef CONFIG_TFL_USE_DEPTH_TO_SPACE
    1 +
#endif
#ifdef CONFIG_TFL_USE_DEPTHWISE_CONV_2D
    1 +
#endif
#ifdef CONFIG_TFL_USE_DEQUANTIZE
    1 +
#endif
#ifdef CONFIG_TFL_USE_DETECTION_POSTPROCESS
    1 +
#endif
#ifdef CONFIG_TFL_USE_DIV
    1 +
#endif
#ifdef CONFIG_TFL_USE_DYNAMIC_UPDATE_SLICE
    1 +
#endif
#ifdef CONFIG_TFL_USE_EMBEDDING_LOOKUP
    1 +
#endif
#ifdef CONFIG_TFL_USE_ENERGY
    1 +
#endif
#ifdef CONFIG_TFL_USE_ELU
    1 +
#endif
#ifdef CONFIG_TFL_USE_EQUAL
    1 +
#endif
#ifdef CONFIG_TFL_USE_ETHOSU
    1 +
#endif
#ifdef CONFIG_TFL_USE_EXP
    1 +
#endif
#ifdef CONFIG_TFL_USE_EXPAND_DIMS
    1 +
#endif
#ifdef CONFIG_TFL_USE_FFT_AUTO_SCALE
    1 +
#endif
#ifdef CONFIG_TFL_USE_FILL
    1 +
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK
    1 +
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK_LOG
    1 +
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK_SQUARE_ROOT
    1 +
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK_SPECTRAL_SUBTRACTION
    1 +
#endif
#ifdef CONFIG_TFL_USE_FLOOR
    1 +
#endif
#ifdef CONFIG_TFL_USE_FLOOR_DIV
    1 +
#endif
#ifdef CONFIG_TFL_USE_FLOOR_MOD
    1 +
#endif
#ifdef CONFIG_TFL_USE_FRAMER
    1 +
#endif
#ifdef CONFIG_TFL_USE_FULLY_CONNECTED
    1 +
#endif
#ifdef CONFIG_TFL_USE_GATHER
    1 +
#endif
#ifdef CONFIG_TFL_USE_GATHER_ND
    1 +
#endif
#ifdef CONFIG_TFL_USE_GREATER
    1 +
#endif
#ifdef CONFIG_TFL_USE_GREATER_EQUAL
    1 +
#endif
#ifdef CONFIG_TFL_USE_HARD_SWISH
    1 +
#endif
#ifdef CONFIG_TFL_USE_IF
    1 +
#endif
#ifdef CONFIG_TFL_USE_IRFFT
    1 +
#endif
#ifdef CONFIG_TFL_USE_L2_NORMALIZATION
    1 +
#endif
#ifdef CONFIG_TFL_USE_L2_POOL_2D
    1 +
#endif
#ifdef CONFIG_TFL_USE_LEAKY_RELU
    1 +
#endif
#ifdef CONFIG_TFL_USE_LESS
    1 +
#endif
#ifdef CONFIG_TFL_USE_LESS_EQUAL
    1 +
#endif
#ifdef CONFIG_TFL_USE_LOG
    1 +
#endif
#ifdef CONFIG_TFL_USE_LOGICAL_AND
    1 +
#endif
#ifdef CONFIG_TFL_USE_LOGICAL_NOT
    1 +
#endif
#ifdef CONFIG_TFL_USE_LOGICAL_OR
    1 +
#endif
#ifdef CONFIG_TFL_USE_LOGISTIC
    1 +
#endif
#ifdef CONFIG_TFL_USE_LOG_SOFTMAX
    1 +
#endif
#ifdef CONFIG_TFL_USE_MAXIMUM
    1 +
#endif
#ifdef CONFIG_TFL_USE_MAX_POOL_2D
    1 +
#endif
#ifdef CONFIG_TFL_USE_MIRROR_PAD
    1 +
#endif
#ifdef CONFIG_TFL_USE_MEAN
    1 +
#endif
#ifdef CONFIG_TFL_USE_MINIMUM
    1 +
#endif
#ifdef CONFIG_TFL_USE_MUL
    1 +
#endif
#ifdef CONFIG_TFL_USE_NEG
    1 +
#endif
#ifdef CONFIG_TFL_USE_NOT_EQUAL
    1 +
#endif
#ifdef CONFIG_TFL_USE_OVERLAP_ADD
    1 +
#endif
#ifdef CONFIG_TFL_USE_PACK
    1 +
#endif
#ifdef CONFIG_TFL_USE_PAD
    1 +
#endif
#ifdef CONFIG_TFL_USE_PAD_V2
    1 +
#endif
#ifdef CONFIG_TFL_USE_PCAN
    1 +
#endif
#ifdef CONFIG_TFL_USE_PRELU
    1 +
#endif
#ifdef CONFIG_TFL_USE_QUANTIZE
    1 +
#endif
#ifdef CONFIG_TFL_USE_READ_VARIABLE
    1 +
#endif
#ifdef CONFIG_TFL_USE_REDUCE_ALL
    1 +
#endif
#ifdef CONFIG_TFL_USE_REDUCE_MAX
    1 +
#endif
#ifdef CONFIG_TFL_USE_REDUCE_MIN
    1 +
#endif
#ifdef CONFIG_TFL_USE_RELU
    1 +
#endif
#ifdef CONFIG_TFL_USE_RELU6
    1 +
#endif
#ifdef CONFIG_TFL_USE_RESHAPE
    1 +
#endif
#ifdef CONFIG_TFL_USE_RESIZE_BILINEAR
    1 +
#endif
#ifdef CONFIG_TFL_USE_RESIZE_NEAREST_NEIGHBOR
    1 +
#endif
#ifdef CONFIG_TFL_USE_REVERSE_V2
    1 +
#endif
#ifdef CONFIG_TFL_USE_RFFT
    1 +
#endif
#ifdef CONFIG_TFL_USE_ROUND
    1 +
#endif
#ifdef CONFIG_TFL_USE_RSQRT
    1 +
#endif
#ifdef CONFIG_TFL_USE_SELECT_V2
    1 +
#endif
#ifdef CONFIG_TFL_USE_SHAPE
    1 +
#endif
#ifdef CONFIG_TFL_USE_SIN
    1 +
#endif
#ifdef CONFIG_TFL_USE_SLICE
    1 +
#endif
#ifdef CONFIG_TFL_USE_SOFTMAX
    1 +
#endif
#ifdef CONFIG_TFL_USE_SPACE_TO_BATCH_ND
    1 +
#endif
#ifdef CONFIG_TFL_USE_SPACE_TO_DEPTH
    1 +
#endif
#ifdef CONFIG_TFL_USE_SPLIT
    1 +
#endif
#ifdef CONFIG_TFL_USE_SPLIT_V
    1 +
#endif
#ifdef CONFIG_TFL_USE_SQUEEZE
    1 +
#endif
#ifdef CONFIG_TFL_USE_SQRT
    1 +
#endif
#ifdef CONFIG_TFL_USE_SQUARE
    1 +
#endif
#ifdef CONFIG_TFL_USE_SQUARED_DIFFERENCE
    1 +
#endif
#ifdef CONFIG_TFL_USE_STRIDED_SLICE
    1 +
#endif
#ifdef CONFIG_TFL_USE_STACKER
    1 +
#endif
#ifdef CONFIG_TFL_USE_SUB
    1 +
#endif
#ifdef CONFIG_TFL_USE_SUM
    1 +
#endif
#ifdef CONFIG_TFL_USE_SVDF
    1 +
#endif
#ifdef CONFIG_TFL_USE_TANH
    1 +
#endif
#ifdef CONFIG_TFL_USE_TRANSPOSE_CONV
    1 +
#endif
#ifdef CONFIG_TFL_USE_TRANSPOSE
    1 +
#endif
#ifdef CONFIG_TFL_USE_UNPACK
    1 +
#endif
#ifdef CONFIG_TFL_USE_UNIDIRECTIONAL_SEQUENCE_LSTM
    1 +
#endif
#ifdef CONFIG_TFL_USE_VAR_HANDLE
    1 +
#endif
#ifdef CONFIG_TFL_USE_WHILE
    1 +
#endif
#ifdef CONFIG_TFL_USE_WINDOW
    1 +
#endif
#ifdef CONFIG_TFL_USE_ZEROS_LIKE
    1 +
#endif
    0;

constexpr unsigned int kOpResolverSize = (kMaxOpCount > 0) ? kMaxOpCount : 1;

template <typename Resolver>
inline void register_configured_ops(Resolver& resolver) {
#ifdef CONFIG_TFL_USE_ABS
    resolver.AddAbs();
#endif
#ifdef CONFIG_TFL_USE_ADD
    resolver.AddAdd();
#endif
#ifdef CONFIG_TFL_USE_ADD_N
    resolver.AddAddN();
#endif
#ifdef CONFIG_TFL_USE_ARG_MAX
    resolver.AddArgMax();
#endif
#ifdef CONFIG_TFL_USE_ARG_MIN
    resolver.AddArgMin();
#endif
#ifdef CONFIG_TFL_USE_ASSIGN_VARIABLE
    resolver.AddAssignVariable();
#endif
#ifdef CONFIG_TFL_USE_AVERAGE_POOL_2D
    resolver.AddAveragePool2D();
#endif
#ifdef CONFIG_TFL_USE_BATCH_MATMUL
    resolver.AddBatchMatMul();
#endif
#ifdef CONFIG_TFL_USE_BATCH_TO_SPACE_ND
    resolver.AddBatchToSpaceNd();
#endif
#ifdef CONFIG_TFL_USE_BROADCAST_ARGS
    resolver.AddBroadcastArgs();
#endif
#ifdef CONFIG_TFL_USE_BROADCAST_TO
    resolver.AddBroadcastTo();
#endif
#ifdef CONFIG_TFL_USE_CALL_ONCE
    resolver.AddCallOnce();
#endif
#ifdef CONFIG_TFL_USE_CAST
    resolver.AddCast();
#endif
#ifdef CONFIG_TFL_USE_CEIL
    resolver.AddCeil();
#endif
#ifdef CONFIG_TFL_USE_CIRCULAR_BUFFER
    resolver.AddCircularBuffer();
#endif
#ifdef CONFIG_TFL_USE_CONCATENATION
    resolver.AddConcatenation();
#endif
#ifdef CONFIG_TFL_USE_CONV_2D
    resolver.AddConv2D();
#endif
#ifdef CONFIG_TFL_USE_COS
    resolver.AddCos();
#endif
#ifdef CONFIG_TFL_USE_CUMSUM
    resolver.AddCumSum();
#endif
#ifdef CONFIG_TFL_USE_DECODE
    resolver.AddDecode();
#endif
#ifdef CONFIG_TFL_USE_DELAY
    resolver.AddDelay();
#endif
#ifdef CONFIG_TFL_USE_DEPTH_TO_SPACE
    resolver.AddDepthToSpace();
#endif
#ifdef CONFIG_TFL_USE_DEPTHWISE_CONV_2D
    resolver.AddDepthwiseConv2D();
#endif
#ifdef CONFIG_TFL_USE_DEQUANTIZE
    resolver.AddDequantize();
#endif
#ifdef CONFIG_TFL_USE_DETECTION_POSTPROCESS
    resolver.AddDetectionPostprocess();
#endif
#ifdef CONFIG_TFL_USE_DIV
    resolver.AddDiv();
#endif
#ifdef CONFIG_TFL_USE_DYNAMIC_UPDATE_SLICE
    resolver.AddDynamicUpdateSlice();
#endif
#ifdef CONFIG_TFL_USE_EMBEDDING_LOOKUP
    resolver.AddEmbeddingLookup();
#endif
#ifdef CONFIG_TFL_USE_ENERGY
    resolver.AddEnergy();
#endif
#ifdef CONFIG_TFL_USE_ELU
    resolver.AddElu();
#endif
#ifdef CONFIG_TFL_USE_EQUAL
    resolver.AddEqual();
#endif
#ifdef CONFIG_TFL_USE_ETHOSU
    resolver.AddEthosU();
#endif
#ifdef CONFIG_TFL_USE_EXP
    resolver.AddExp();
#endif
#ifdef CONFIG_TFL_USE_EXPAND_DIMS
    resolver.AddExpandDims();
#endif
#ifdef CONFIG_TFL_USE_FFT_AUTO_SCALE
    resolver.AddFftAutoScale();
#endif
#ifdef CONFIG_TFL_USE_FILL
    resolver.AddFill();
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK
    resolver.AddFilterBank();
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK_LOG
    resolver.AddFilterBankLog();
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK_SQUARE_ROOT
    resolver.AddFilterBankSquareRoot();
#endif
#ifdef CONFIG_TFL_USE_FILTER_BANK_SPECTRAL_SUBTRACTION
    resolver.AddFilterBankSpectralSubtraction();
#endif
#ifdef CONFIG_TFL_USE_FLOOR
    resolver.AddFloor();
#endif
#ifdef CONFIG_TFL_USE_FLOOR_DIV
    resolver.AddFloorDiv();
#endif
#ifdef CONFIG_TFL_USE_FLOOR_MOD
    resolver.AddFloorMod();
#endif
#ifdef CONFIG_TFL_USE_FRAMER
    resolver.AddFramer();
#endif
#ifdef CONFIG_TFL_USE_FULLY_CONNECTED
    resolver.AddFullyConnected();
#endif
#ifdef CONFIG_TFL_USE_GATHER
    resolver.AddGather();
#endif
#ifdef CONFIG_TFL_USE_GATHER_ND
    resolver.AddGatherNd();
#endif
#ifdef CONFIG_TFL_USE_GREATER
    resolver.AddGreater();
#endif
#ifdef CONFIG_TFL_USE_GREATER_EQUAL
    resolver.AddGreaterEqual();
#endif
#ifdef CONFIG_TFL_USE_HARD_SWISH
    resolver.AddHardSwish();
#endif
#ifdef CONFIG_TFL_USE_IF
    resolver.AddIf();
#endif
#ifdef CONFIG_TFL_USE_IRFFT
    resolver.AddIrfft();
#endif
#ifdef CONFIG_TFL_USE_L2_NORMALIZATION
    resolver.AddL2Normalization();
#endif
#ifdef CONFIG_TFL_USE_L2_POOL_2D
    resolver.AddL2Pool2D();
#endif
#ifdef CONFIG_TFL_USE_LEAKY_RELU
    resolver.AddLeakyRelu();
#endif
#ifdef CONFIG_TFL_USE_LESS
    resolver.AddLess();
#endif
#ifdef CONFIG_TFL_USE_LESS_EQUAL
    resolver.AddLessEqual();
#endif
#ifdef CONFIG_TFL_USE_LOG
    resolver.AddLog();
#endif
#ifdef CONFIG_TFL_USE_LOGICAL_AND
    resolver.AddLogicalAnd();
#endif
#ifdef CONFIG_TFL_USE_LOGICAL_NOT
    resolver.AddLogicalNot();
#endif
#ifdef CONFIG_TFL_USE_LOGICAL_OR
    resolver.AddLogicalOr();
#endif
#ifdef CONFIG_TFL_USE_LOGISTIC
    resolver.AddLogistic();
#endif
#ifdef CONFIG_TFL_USE_LOG_SOFTMAX
    resolver.AddLogSoftmax();
#endif
#ifdef CONFIG_TFL_USE_MAXIMUM
    resolver.AddMaximum();
#endif
#ifdef CONFIG_TFL_USE_MAX_POOL_2D
    resolver.AddMaxPool2D();
#endif
#ifdef CONFIG_TFL_USE_MIRROR_PAD
    resolver.AddMirrorPad();
#endif
#ifdef CONFIG_TFL_USE_MEAN
    resolver.AddMean();
#endif
#ifdef CONFIG_TFL_USE_MINIMUM
    resolver.AddMinimum();
#endif
#ifdef CONFIG_TFL_USE_MUL
    resolver.AddMul();
#endif
#ifdef CONFIG_TFL_USE_NEG
    resolver.AddNeg();
#endif
#ifdef CONFIG_TFL_USE_NOT_EQUAL
    resolver.AddNotEqual();
#endif
#ifdef CONFIG_TFL_USE_OVERLAP_ADD
    resolver.AddOverlapAdd();
#endif
#ifdef CONFIG_TFL_USE_PACK
    resolver.AddPack();
#endif
#ifdef CONFIG_TFL_USE_PAD
    resolver.AddPad();
#endif
#ifdef CONFIG_TFL_USE_PAD_V2
    resolver.AddPadV2();
#endif
#ifdef CONFIG_TFL_USE_PCAN
    resolver.AddPCAN();
#endif
#ifdef CONFIG_TFL_USE_PRELU
    resolver.AddPrelu();
#endif
#ifdef CONFIG_TFL_USE_QUANTIZE
    resolver.AddQuantize();
#endif
#ifdef CONFIG_TFL_USE_READ_VARIABLE
    resolver.AddReadVariable();
#endif
#ifdef CONFIG_TFL_USE_REDUCE_ALL
    resolver.AddReduceAll();
#endif
#ifdef CONFIG_TFL_USE_REDUCE_MAX
    resolver.AddReduceMax();
#endif
#ifdef CONFIG_TFL_USE_REDUCE_MIN
    resolver.AddReduceMin();
#endif
#ifdef CONFIG_TFL_USE_RELU
    resolver.AddRelu();
#endif
#ifdef CONFIG_TFL_USE_RELU6
    resolver.AddRelu6();
#endif
#ifdef CONFIG_TFL_USE_RESHAPE
    resolver.AddReshape();
#endif
#ifdef CONFIG_TFL_USE_RESIZE_BILINEAR
    resolver.AddResizeBilinear();
#endif
#ifdef CONFIG_TFL_USE_RESIZE_NEAREST_NEIGHBOR
    resolver.AddResizeNearestNeighbor();
#endif
#ifdef CONFIG_TFL_USE_REVERSE_V2
    resolver.AddReverseV2();
#endif
#ifdef CONFIG_TFL_USE_RFFT
    resolver.AddRfft();
#endif
#ifdef CONFIG_TFL_USE_ROUND
    resolver.AddRound();
#endif
#ifdef CONFIG_TFL_USE_RSQRT
    resolver.AddRsqrt();
#endif
#ifdef CONFIG_TFL_USE_SELECT_V2
    resolver.AddSelectV2();
#endif
#ifdef CONFIG_TFL_USE_SHAPE
    resolver.AddShape();
#endif
#ifdef CONFIG_TFL_USE_SIN
    resolver.AddSin();
#endif
#ifdef CONFIG_TFL_USE_SLICE
    resolver.AddSlice();
#endif
#ifdef CONFIG_TFL_USE_SOFTMAX
    resolver.AddSoftmax();
#endif
#ifdef CONFIG_TFL_USE_SPACE_TO_BATCH_ND
    resolver.AddSpaceToBatchNd();
#endif
#ifdef CONFIG_TFL_USE_SPACE_TO_DEPTH
    resolver.AddSpaceToDepth();
#endif
#ifdef CONFIG_TFL_USE_SPLIT
    resolver.AddSplit();
#endif
#ifdef CONFIG_TFL_USE_SPLIT_V
    resolver.AddSplitV();
#endif
#ifdef CONFIG_TFL_USE_SQUEEZE
    resolver.AddSqueeze();
#endif
#ifdef CONFIG_TFL_USE_SQRT
    resolver.AddSqrt();
#endif
#ifdef CONFIG_TFL_USE_SQUARE
    resolver.AddSquare();
#endif
#ifdef CONFIG_TFL_USE_SQUARED_DIFFERENCE
    resolver.AddSquaredDifference();
#endif
#ifdef CONFIG_TFL_USE_STRIDED_SLICE
    resolver.AddStridedSlice();
#endif
#ifdef CONFIG_TFL_USE_STACKER
    resolver.AddStacker();
#endif
#ifdef CONFIG_TFL_USE_SUB
    resolver.AddSub();
#endif
#ifdef CONFIG_TFL_USE_SUM
    resolver.AddSum();
#endif
#ifdef CONFIG_TFL_USE_SVDF
    resolver.AddSvdf();
#endif
#ifdef CONFIG_TFL_USE_TANH
    resolver.AddTanh();
#endif
#ifdef CONFIG_TFL_USE_TRANSPOSE_CONV
    resolver.AddTransposeConv();
#endif
#ifdef CONFIG_TFL_USE_TRANSPOSE
    resolver.AddTranspose();
#endif
#ifdef CONFIG_TFL_USE_UNPACK
    resolver.AddUnpack();
#endif
#ifdef CONFIG_TFL_USE_UNIDIRECTIONAL_SEQUENCE_LSTM
    resolver.AddUnidirectionalSequenceLSTM();
#endif
#ifdef CONFIG_TFL_USE_VAR_HANDLE
    resolver.AddVarHandle();
#endif
#ifdef CONFIG_TFL_USE_WHILE
    resolver.AddWhile();
#endif
#ifdef CONFIG_TFL_USE_WINDOW
    resolver.AddWindow();
#endif
#ifdef CONFIG_TFL_USE_ZEROS_LIKE
    resolver.AddZerosLike();
#endif
}

#endif /* __cplusplus */

#endif /* TFLITE_OPS_HPP */