/* -*- c++ -*- */
/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (C) 2024 SDR-LoRa contributors.
 *
 * Part of SDR-LoRa: https://github.com/fabio-busacca/sdr-lora
 * See LICENSE for the full text. */
#ifndef INCLUDED_SDRLORA_API_H
#define INCLUDED_SDRLORA_API_H
#include <gnuradio/attributes.h>
#ifdef gnuradio_sdrlora_EXPORTS
#define SDRLORA_API __GR_ATTR_EXPORT
#else
#define SDRLORA_API __GR_ATTR_IMPORT
#endif
#endif
