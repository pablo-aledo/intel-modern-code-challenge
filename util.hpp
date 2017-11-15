/*
  Copyright (c) 2015, Newcastle University (United Kingdom)
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <cstdio>
#include <vector>

struct cdc_params
{
    float    speed;
    unsigned have_speed;
    int64_t  T;
    unsigned have_T;
    int64_t  L;
    unsigned have_L;
    float    D;
    unsigned have_D;
    float    mu;
    unsigned have_mu;
    unsigned divThreshold;
    unsigned have_divThreshold;
    float    spatialScale;
    unsigned have_spatialScale;
    float    pathThreshold;
    unsigned have_pathThreshold;
    int64_t  finalNumberCells;
    float    spatialRange;
};

struct stopwatch
{
    stopwatch()
    {
        elapsed = 0.0;
        count = 0;
    }

    void reset()
    {
        clock_gettime(CLOCK_MONOTONIC, &last);
    }

    double average() const
    {
        return elapsed/count;
    }

    double mark()
    {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        const double interval = (now.tv_sec-last.tv_sec + 1e-9*(now.tv_nsec - last.tv_nsec));
        elapsed += interval;
        last = now;
        ++count;
        return interval;
    }
    struct timespec last;

    int64_t count;
    double elapsed;
};

char *read_kv(char *argv[], int in_ind, int *optind);

cdc_params get_params(const char *input_file, std::vector<char*> &candidate_kvs, int quiet);

void print_params(const cdc_params *p, FILE *out);

void die(const char *fmt, ...);

void print_sys_config(FILE *o);
