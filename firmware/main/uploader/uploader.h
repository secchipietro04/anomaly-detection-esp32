#ifndef UPLOADER_H
#define UPLOADER_H

#include "quad_buffer.h"

#define TELEMETRY_CHUNK_SIZE    512

// start the network uploader task
void uploader_start(quad_buffer_t *qb);

#endif // UPLOADER_H
