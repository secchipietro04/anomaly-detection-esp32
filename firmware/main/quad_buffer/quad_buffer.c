#include "quad_buffer.h"
#include <string.h>

QueueHandle_t free_slots_queue = NULL;
QueueHandle_t ready_inf_queue = NULL;
QueueHandle_t pending_sd_queue = NULL;
QueueHandle_t ready_send_queue = NULL;

void quad_buffer_init(quad_buffer_t *qb) {
    // reset indices
    qb->write_idx = 0;
    qb->process_idx = 0;
    
    // init each slot
    for (int i = 0; i < 4; i++) {
        qb->slots[i].flags = BUF_FLAG_FREE;
        qb->slots[i].segment_id = 0;
        qb->slots[i].sample_rate = 3840.0f;
        qb->slots[i].is_inferenced = false;
        qb->slots[i].is_anomaly = false;
        qb->slots[i].is_replay = false;
        qb->slots[i].anomaly_score = 0.0f;
        
        // create mutex for flag updates
        qb->slots[i].mutex = xSemaphoreCreateMutex();
    }
    
    free_slots_queue = xQueueCreate(4, sizeof(int));
    ready_inf_queue = xQueueCreate(4, sizeof(int));
    pending_sd_queue = xQueueCreate(4, sizeof(int));
    ready_send_queue = xQueueCreate(4, sizeof(int));
    
    // initially, all slots are free
    for (int i = 0; i < 4; i++) {
        int idx = i;
        xQueueSend(free_slots_queue, &idx, 0);
    }
}

void quad_buffer_lock(buffer_slot_t *slot) {
    if (slot->mutex) {
        xSemaphoreTake(slot->mutex, portMAX_DELAY);
    }
}

void quad_buffer_unlock(buffer_slot_t *slot) {
    if (slot->mutex) {
        xSemaphoreGive(slot->mutex);
    }
}
