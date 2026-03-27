#include "cjson.h"
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>

typedef struct {
    const char *s;
    size_t pos;
} parser_t;

static void skip_ws(parser_t *p) {
    while (p->s[p->pos] && isspace((unsigned char)p->s[p->pos])) p->pos++;
}

static json_value_t *alloc_value(json_type_t type) {
    json_value_t *v = calloc(1, sizeof(json_value_t));
    if (v) v->type = type;
    return v;
}

static json_value_t *parse_value(parser_t *p);

static char *parse_string_raw(parser_t *p) {
    if (p->s[p->pos] != '"') return NULL;
    p->pos++;
    size_t start = p->pos;
    /* First pass: compute length */
    size_t len = 0;
    size_t i = start;
    while (p->s[i] && p->s[i] != '"') {
        if (p->s[i] == '\\') {
            i++;
            if (p->s[i] == 'u') { i += 4; }
        }
        len++;
        i++;
    }

    char *buf = malloc(len + 1);
    if (!buf) return NULL;
    size_t j = 0;
    while (p->s[p->pos] && p->s[p->pos] != '"') {
        if (p->s[p->pos] == '\\') {
            p->pos++;
            switch (p->s[p->pos]) {
                case '"':  buf[j++] = '"'; break;
                case '\\': buf[j++] = '\\'; break;
                case '/':  buf[j++] = '/'; break;
                case 'b':  buf[j++] = '\b'; break;
                case 'f':  buf[j++] = '\f'; break;
                case 'n':  buf[j++] = '\n'; break;
                case 'r':  buf[j++] = '\r'; break;
                case 't':  buf[j++] = '\t'; break;
                case 'u': {
                    /* Parse unicode escape - handle BMP only */
                    char hex[5] = {0};
                    memcpy(hex, &p->s[p->pos + 1], 4);
                    unsigned int cp = (unsigned int)strtoul(hex, NULL, 16);
                    p->pos += 4;
                    if (cp < 0x80) {
                        buf[j++] = (char)cp;
                    } else if (cp < 0x800) {
                        buf = realloc(buf, len + 4);
                        buf[j++] = (char)(0xC0 | (cp >> 6));
                        buf[j++] = (char)(0x80 | (cp & 0x3F));
                    } else {
                        buf = realloc(buf, len + 4);
                        buf[j++] = (char)(0xE0 | (cp >> 12));
                        buf[j++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        buf[j++] = (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: buf[j++] = p->s[p->pos]; break;
            }
        } else {
            buf[j++] = p->s[p->pos];
        }
        p->pos++;
    }
    buf[j] = '\0';
    if (p->s[p->pos] == '"') p->pos++;
    return buf;
}

static json_value_t *parse_string(parser_t *p) {
    char *s = parse_string_raw(p);
    if (!s) return NULL;
    json_value_t *v = alloc_value(JSON_STRING);
    v->string = s;
    return v;
}

static json_value_t *parse_number(parser_t *p) {
    char *end;
    double num = strtod(&p->s[p->pos], &end);
    if (end == &p->s[p->pos]) return NULL;
    p->pos = (size_t)(end - p->s);
    json_value_t *v = alloc_value(JSON_NUMBER);
    v->number = num;
    return v;
}

static json_value_t *parse_array(parser_t *p) {
    p->pos++; /* skip '[' */
    json_value_t *v = alloc_value(JSON_ARRAY);
    v->array = NULL;
    json_element_t **tail = &v->array;

    skip_ws(p);
    if (p->s[p->pos] == ']') { p->pos++; return v; }

    while (1) {
        skip_ws(p);
        json_value_t *elem = parse_value(p);
        if (!elem) { json_free(v); return NULL; }

        json_element_t *el = calloc(1, sizeof(json_element_t));
        el->value = elem;
        *tail = el;
        tail = &el->next;

        skip_ws(p);
        if (p->s[p->pos] == ',') { p->pos++; continue; }
        if (p->s[p->pos] == ']') { p->pos++; break; }
        json_free(v);
        return NULL;
    }
    return v;
}

static json_value_t *parse_object(parser_t *p) {
    p->pos++; /* skip '{' */
    json_value_t *v = alloc_value(JSON_OBJECT);
    v->object = NULL;
    json_member_t **tail = &v->object;

    skip_ws(p);
    if (p->s[p->pos] == '}') { p->pos++; return v; }

    while (1) {
        skip_ws(p);
        char *key = parse_string_raw(p);
        if (!key) { json_free(v); return NULL; }

        skip_ws(p);
        if (p->s[p->pos] != ':') { free(key); json_free(v); return NULL; }
        p->pos++;

        skip_ws(p);
        json_value_t *val = parse_value(p);
        if (!val) { free(key); json_free(v); return NULL; }

        json_member_t *m = calloc(1, sizeof(json_member_t));
        m->key = key;
        m->value = val;
        *tail = m;
        tail = &m->next;

        skip_ws(p);
        if (p->s[p->pos] == ',') { p->pos++; continue; }
        if (p->s[p->pos] == '}') { p->pos++; break; }
        json_free(v);
        return NULL;
    }
    return v;
}

static json_value_t *parse_value(parser_t *p) {
    skip_ws(p);
    char c = p->s[p->pos];

    if (c == '"') return parse_string(p);
    if (c == '{') return parse_object(p);
    if (c == '[') return parse_array(p);
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(p);
    if (strncmp(&p->s[p->pos], "true", 4) == 0) {
        p->pos += 4;
        json_value_t *v = alloc_value(JSON_BOOL);
        v->boolean = 1;
        return v;
    }
    if (strncmp(&p->s[p->pos], "false", 5) == 0) {
        p->pos += 5;
        json_value_t *v = alloc_value(JSON_BOOL);
        v->boolean = 0;
        return v;
    }
    if (strncmp(&p->s[p->pos], "null", 4) == 0) {
        p->pos += 4;
        return alloc_value(JSON_NULL);
    }
    return NULL;
}

json_value_t *json_parse(const char *text) {
    parser_t p = { .s = text, .pos = 0 };
    return parse_value(&p);
}

void json_free(json_value_t *v) {
    if (!v) return;
    switch (v->type) {
        case JSON_STRING:
            free(v->string);
            break;
        case JSON_ARRAY: {
            json_element_t *el = v->array;
            while (el) {
                json_element_t *next = el->next;
                json_free(el->value);
                free(el);
                el = next;
            }
            break;
        }
        case JSON_OBJECT: {
            json_member_t *m = v->object;
            while (m) {
                json_member_t *next = m->next;
                free(m->key);
                json_free(m->value);
                free(m);
                m = next;
            }
            break;
        }
        default:
            break;
    }
    free(v);
}

json_value_t *json_object_get(const json_value_t *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (json_member_t *m = obj->object; m; m = m->next) {
        if (strcmp(m->key, key) == 0) return m->value;
    }
    return NULL;
}

size_t json_array_length(const json_value_t *arr) {
    if (!arr || arr->type != JSON_ARRAY) return 0;
    size_t n = 0;
    for (json_element_t *el = arr->array; el; el = el->next) n++;
    return n;
}

json_value_t *json_array_get(const json_value_t *arr, size_t index) {
    if (!arr || arr->type != JSON_ARRAY) return NULL;
    json_element_t *el = arr->array;
    for (size_t i = 0; i < index && el; i++) el = el->next;
    return el ? el->value : NULL;
}

const char *json_string_value(const json_value_t *v) {
    return (v && v->type == JSON_STRING) ? v->string : NULL;
}

double json_number_value(const json_value_t *v) {
    return (v && v->type == JSON_NUMBER) ? v->number : 0.0;
}
