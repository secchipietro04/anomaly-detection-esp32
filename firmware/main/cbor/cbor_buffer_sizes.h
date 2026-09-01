#ifndef CBOR_BUFFER_SIZES_H
#define CBOR_BUFFER_SIZES_H

#include "cbor/gen/telemetry_types.h"
#include "cbor/gen/device_types.h"
#include "cbor/gen/ensemble_types.h"
#include <assert.h>

// expansion factor for 32-bit float values in standard CBOR (1-byte 0xFA header + 4-byte payload = 5 bytes)
#define CBOR_FLOAT_EXPANSION_RATIO(bytes) (((bytes) * 5) / 4)

// safety margin for CBOR container tags, maps, string keys and scalar fields
#define CBOR_HEADER_MARGIN_BYTES 512

// automatic compile-time maximum encoded buffer size for telemetry Segment packets
#define CBOR_SEGMENT_MAX_ENCODED_LEN ( \
    CBOR_FLOAT_EXPANSION_RATIO(sizeof(((struct Segment *)0)->data_gyro) + \
                               sizeof(((struct Segment *)0)->data_accel)) + \
    CBOR_HEADER_MARGIN_BYTES )

// standard buffer size for small diagnostic / scalar CBOR payloads
#define CBOR_SMALL_PAYLOAD_MAX_LEN 512

// compile-time verification
_Static_assert(CBOR_SEGMENT_MAX_ENCODED_LEN >= 15360, "CBOR segment buffer calculation smaller than data payload minimum");

#endif // CBOR_BUFFER_SIZES_H
