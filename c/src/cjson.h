#ifndef CJSON_H
#define CJSON_H

#include <stddef.h>

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type_t;

typedef struct json_value json_value_t;

typedef struct json_member {
    char *key;
    json_value_t *value;
    struct json_member *next;
} json_member_t;

typedef struct json_element {
    json_value_t *value;
    struct json_element *next;
} json_element_t;

struct json_value {
    json_type_t type;
    union {
        int boolean;
        double number;
        char *string;
        json_element_t *array;
        json_member_t *object;
    };
};

/* Parse JSON string, returns NULL on error */
json_value_t *json_parse(const char *text);

/* Free parsed JSON tree */
void json_free(json_value_t *v);

/* Access helpers */
json_value_t *json_object_get(const json_value_t *obj, const char *key);
size_t json_array_length(const json_value_t *arr);
json_value_t *json_array_get(const json_value_t *arr, size_t index);

/* Convenience */
const char *json_string_value(const json_value_t *v);
double json_number_value(const json_value_t *v);

#endif
