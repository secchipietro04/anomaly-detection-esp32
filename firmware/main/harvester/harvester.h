#ifndef HARVESTER_H
#define HARVESTER_H

#include "quad_buffer.h"

// start the harvester task
void harvester_start(quad_buffer_t *qb);

// update sample rate at runtime
void harvester_update_rate(float rate_hz);

#endif // HARVESTER_H
