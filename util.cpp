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

#include <gnu/libc-version.h>
#include <sys/utsname.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>
#include <cstdarg>
#include <cmath>

#include "util.hpp"

void die(const char *fmt, ...)
{
    va_list val;
    va_start(val, fmt);
    vfprintf(stderr, fmt, val);
    va_end(val);
    exit(EXIT_FAILURE);
}

static char *format_uname()
{
    utsname un;
    if(uname(&un) == -1)
    {
        perror("uname");
        return 0;
    }
    char buff[1024];
    snprintf(buff, 1023, "%s-%s-%s-%s", un.nodename, un.machine, un.sysname, un.release);
    return strdup(buff);
}

static void cpuid(const unsigned info, unsigned *eax, unsigned *ebx, unsigned *ecx, unsigned *edx)
{
    __asm__("cpuid;"
            :"=a" (*eax), "=b" (*ebx), "=c" (*ecx), "=d" (*edx)
            :"a" (info));
}

static void vendorid(char id[13])
{
    unsigned temp;
    cpuid(0, &temp, (unsigned *)id, (unsigned *)(id+8), (unsigned *)(id+4));
    id[12] = 0;
}

static void proc_brand(char str[49])
{
    static char nope[] = "Unknown";
    unsigned okay;
    cpuid(0x80000000, &okay, (unsigned *)str, (unsigned *)(str+4), (unsigned *)(str+8));
    if(okay < 0x80000004)
        strcpy(str, nope);
    else
    {
        for(unsigned i = 0; i < 3; ++i)
            cpuid(0x80000002+i, (unsigned *)(str+16*i), (unsigned *)(str+16*i+4), (unsigned *)(str+16*i+8), (unsigned *)(str+16*i+12));
    }
    str[48] = 0;
}

struct cpuinfo
{
    unsigned stepping : 4;
    unsigned model : 4;
    unsigned family_id : 4;
    unsigned proc_type : 2;
    unsigned nothing : 2;
    unsigned extended_model_id : 4;
    unsigned extended_family_id : 8;
    unsigned nothing2 : 6;
    unsigned display_family;
    unsigned display_model;
};

static void cpu_info(cpuinfo *ci)
{
    unsigned b, c, d;
    cpuid(0x1, (unsigned *)ci, &b, &c, &d);
    if(ci->family_id == 0x0F)
        ci->display_family = ci->extended_family_id + ci->family_id;
    else
        ci->display_family = ci->family_id;
    if(ci->family_id == 0x0F || ci->family_id == 0x06)
        ci->display_model = (ci->extended_model_id << 4) + ci->model;
    else
        ci->display_model = ci->model;
}

void print_sys_config(FILE *o)
{
    fprintf(o, "%-35s = %s\n", "BUILD_HOST", BUILD_HOST);
    fprintf(o, "%-35s = %s\n", "COMPILER_VERSION", COMPILER_VERSION);
    fprintf(o, "%-35s = %s\n", "GLIBC_VERSION", gnu_get_libc_version ());
    fprintf(o, "%-35s = %s %s\n", "BUILD_DATE", __DATE__, __TIME__);
    char *host = format_uname();
    fprintf(o, "%-35s = %s\n", "HOST", host);
    free(host);
    char vid[13];
    vendorid(vid);
    char pb[49];
    proc_brand(pb);
    fprintf(o, "%-35s = %s %s\n", "CPU", vid, pb);
    cpuinfo ci;
    cpu_info(&ci);
    fprintf(o, "%-35s = %s\n", "LD_PRELOAD", getenv("LD_PRELOAD"));
    fprintf(o, "%-35s = Family %u Model %u Stepping %u\n", "CPUINFO", ci.display_family, ci.display_model, ci.stepping);
    fprintf(o, "%-35s = %s\n", "KMP_AFFINITY", getenv("KMP_AFFINITY"));
    fprintf(o, "%-35s = %s\n", "KMP_BLOCKTIME", getenv("KMP_BLOCKTIME"));
}

char *read_kv(char **argv, int in_ind, int *optind)
{
    char *nondash = argv[in_ind];
    while(*nondash && *nondash == '-')
        ++nondash;

    char *kvstr;
    int   klen = strlen(nondash);
    if(!strchr(nondash, '='))
    {
        const int vlen = argv[in_ind+1] ? strlen(argv[in_ind+1]) : 0;
        kvstr = (char*)malloc(klen + 1 + vlen + 1);
        strcpy(kvstr, nondash);
        kvstr[klen] = '=';
        if(vlen)
        {
            strcpy(kvstr + klen + 1, argv[in_ind+1]);
            ++*optind;
        }
    }
    else
        kvstr = strdup(nondash);

    return kvstr;
}

static void keyval(char *buffer, char **pkey, char **pval)
{
    char *ptr;
    *pkey = buffer;
    *pval = buffer;

    // kill the newline
    *pval = strchr(buffer, '\n');
    if (*pval)
        **pval = 0;

    // suppress leading whites or tabs
    while ((**pkey == ' ') || (**pkey == '\t'))
        (*pkey)++;
    *pval = strchr(buffer, '=');
    if (*pval) {
        **pval = 0;
        (*pval)++;
    }
    // strip key from white or tab
    while ((ptr = strchr(*pkey, ' ')) != NULL) {
        *ptr = 0;
    }
    while ((ptr = strchr(*pkey, '\t')) != NULL) {
        *ptr = 0;
    }
}

static bool cdc_set_kv(cdc_params *params, char *kvstr, unsigned stage)
{
    char *pkey, *pval;
    keyval(kvstr, &pkey, &pval);

    if (strcmp(pkey, "speed") == 0) {
        if(params->have_speed >= stage)
            die("Found duplicate speed!");
        sscanf(pval, "%f", &params->speed);
        params->have_speed = stage;
        return true;
    }
    if (strcmp(pkey, "T") == 0) {
        if(params->have_T >= stage)
            die("Found duplicate T!");
        sscanf(pval, "%lld", (long long int*)&params->T);
        params->have_T = stage;
        return true;
    }
    if (strcmp(pkey, "L") == 0) {
        if(params->have_L >= stage)
            die("Found duplicate L!");
        sscanf(pval, "%lld", (long long int*)&params->L);
        params->have_L = stage;
        return true;
    }
    if (strcmp(pkey, "D") == 0) {
        if(params->have_D >= stage)
            die("Found duplicate D!");
        sscanf(pval, "%f", &params->D);
        params->have_D = stage;
        return true;
    }
    if (strcmp(pkey, "mu") == 0) {
        if(params->have_mu >= stage)
            die("Found duplicate mu!");
        sscanf(pval, "%f", &params->mu);
        params->have_mu = stage;
        return true;
    }
    // float parameters
    if (strcmp(pkey, "divThreshold") == 0) {
        if(params->have_divThreshold >= stage)
            die("Found duplicate divThreshold!");
        sscanf(pval, "%u", &params->divThreshold);
        params->have_divThreshold = stage;
        return true;
    }
    if (strcmp(pkey, "pathThreshold") == 0) {
        if(params->have_pathThreshold >= stage)
            die("Found duplicate pathThreshold!");
        sscanf(pval, "%f", &params->pathThreshold);
        params->have_pathThreshold = stage;
        return true;
    }
    if (strcmp(pkey, "spatialScale") == 0) {
        if(params->have_spatialScale >= stage)
            die("Found duplicate spatialScale!");
        sscanf(pval, "%f", &params->spatialScale);
        params->have_spatialScale = stage;
        return true;
    }
    return false;
}

cdc_params get_params(const char *input_file, std::vector<char*> &candidate_kvs, int quiet)
{
    FILE *fp = fopen(input_file, "r");
    if(!fp)
        die("Can't open %s for reading!\n", input_file);

    char buffer[1024];

    cdc_params params;
    params.have_speed         = 0;
    params.have_T             = 0;
    params.have_L             = 0;
    params.have_D             = 0;
    params.have_mu            = 0;
    params.have_divThreshold  = 0;
    params.have_spatialScale  = 0;
    params.have_pathThreshold = 0;

    while (fgets(buffer, 1024, fp) == buffer)
    {
        bool res = cdc_set_kv(&params, buffer, 1);
        if(!res && quiet < 2)
            printf("[PARAMS] Skipping unused key %s\n", buffer);
    }

    fclose(fp);

    for(int64_t i = 0; i < (int64_t)candidate_kvs.size(); ++i)
    {
        if(quiet < 1)
            printf("[setup] Trying kv %s\n", candidate_kvs[i]);

        bool res = cdc_set_kv(&params, candidate_kvs[i], 2);
        if(!res && quiet < 2)
            printf("[setup] Hydro didn't accept option kv %s\n", candidate_kvs[i]);
        free(candidate_kvs[i]);
    }

    if(!params.have_speed)
        die("Missing speed parameter!\n");
    if(!params.have_T)
        die("Missing T parameter!\n");
    if(!params.have_L)
        die("Missing L parameter!\n");
    if(!params.have_D)
        die("Missing D parameter!\n");
    if(!params.have_mu)
        die("Missing mu parameter!\n");
    if(!params.have_divThreshold)
        die("Missing divThreshold parameter!\n");
    if(!params.have_spatialScale)
        die("Missing spatialScale parameter!\n");
    if(!params.have_pathThreshold)
        die("Missing pathThreshold parameter!\n");

    params.finalNumberCells = powf(2.0f,params.divThreshold);
    params.spatialRange     = params.spatialScale*powf(1.0f/((float)(params.finalNumberCells)), 1.0f/3.0f);

    candidate_kvs.clear();
    return params;
}

void print_params(const cdc_params *p, FILE *out)
{
    fprintf(out, "%-35s = %llu\n", "N_INITIAL", 1ULL);
    fprintf(out, "%-35s = %le\n",  "SPEED", p->speed);
    fprintf(out, "%-35s = %llu\n", "T", (long long int)p->T);
    fprintf(out, "%-35s = %llu\n", "L", (long long int)p->L);
    fprintf(out, "%-35s = %le\n",  "D", p->D);
    fprintf(out, "%-35s = %le\n",  "MU", p->mu);
    fprintf(out, "%-35s = %lu\n",  "FINALNUMBERCELLS", p->finalNumberCells);
    fprintf(out, "%-35s = %le\n",  "SPATIALSCALE", p->spatialScale);
    fprintf(out, "%-35s = %le\n",  "SPATIALRANGE", p->spatialRange);
    fprintf(out, "%-35s = %le\n",  "PATHTHRESHOLD", p->pathThreshold);
    fprintf(out, "%-35s = %u\n",   "DIVTHRESHOLD", p->divThreshold);
    fprintf(out, "------------------------------------------\n");
}
