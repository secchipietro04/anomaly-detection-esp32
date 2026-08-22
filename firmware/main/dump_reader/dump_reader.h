#ifndef DUMP_READER_H
#define DUMP_READER_H

#include "quad_buffer.h"

// trigger a historical data dump
void dump_reader_trigger(quad_buffer_t *qb, bool run_inference);



#endif // DUMP_READER_H
