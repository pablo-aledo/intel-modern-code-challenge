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

#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <getopt.h>
#include "util.hpp"

using namespace std;

static int quiet = 0;
int Conc_Pad;
#define Conc(a,b,c,d) Conc[ a*Conc_Pad*Conc_Pad*Conc_Pad + b*Conc_Pad*Conc_Pad + c*Conc_Pad + d ]
#define ping(a,b,c,d) ping[ a*Conc_Pad*Conc_Pad*Conc_Pad + b*Conc_Pad*Conc_Pad + c*Conc_Pad + d ]
#define pong(a,b,c,d) pong[ a*Conc_Pad*Conc_Pad*Conc_Pad + b*Conc_Pad*Conc_Pad + c*Conc_Pad + d ]

static float RandomFloatPos() {
    // returns a random number between a given minimum and maximum
    float random = ((float) rand()) / (float) RAND_MAX;
    float a = 0;
    float r = random;
    return a + r;
}

static float getNorm(float* currArray) {
    // computes L2 norm of input array
    int c;
    float arraySum=0;
    for (c=0; c<3; c++) {
        arraySum += currArray[c]*currArray[c];
    }
    float res = sqrt(arraySum);

    return res;
}

static float getL2Distance(float pos1x, float pos1y, float pos1z, float pos2x, float pos2y, float pos2z) {
    // returns distance (L2 norm) between two positions in 3D
    float distArray[3];
    distArray[0] = pos2x-pos1x;
    distArray[1] = pos2y-pos1y;
    distArray[2] = pos2z-pos1z;
    float l2Norm = getNorm(distArray);
    return l2Norm;
}

static stopwatch produceSubstances_sw;
static stopwatch runDiffusionStep_sw;
static stopwatch runDecayStep_sw;
static stopwatch cellMovementAndDuplication_sw;
static stopwatch runDiffusionClusterStep_sw;
static stopwatch getEnergy_sw;
static stopwatch getCriterion_sw;
static stopwatch extra_sw;

static void produceSubstances(float* Conc, float** posAll, int* typesAll, int L, int n) {
    produceSubstances_sw.reset();

    // increases the concentration of substances at the location of the cells
    float sideLength = 1/(float)L; // length of a side of a diffusion voxel

#pragma omp parallel for
    for (int c=0; c< n; c++) {
        int i1 = std::min((int)floor(posAll[c][0]/sideLength),(L-1));
        int i2 = std::min((int)floor(posAll[c][1]/sideLength),(L-1));
        int i3 = std::min((int)floor(posAll[c][2]/sideLength),(L-1));

        if (typesAll[c]==1) {
            Conc(0,i1,i2,i3)+=0.1;
            if (Conc(0,i1,i2,i3)>1) {
                Conc(0,i1,i2,i3)=1;
            }
        }
        else {
            Conc(1,i1,i2,i3)+=0.1;
            if (Conc(1,i1,i2,i3)>1) {
                Conc(1,i1,i2,i3)=1;
            }
        }
    }
    produceSubstances_sw.mark();
}

static void runDiffusionStep(float* ping, float* pong, int L, float D) {
    runDiffusionStep_sw.reset();
    // computes the changes in substance concentrations due to diffusion
    //int i1,i2,i3;
    //float tempConc(2,L,L,L);
    //for (i1 = 0; i1 < L; i1++) {
        //for (i2 = 0; i2 < L; i2++) {
            //for (i3 = 0; i3 < L; i3++) {
                //tempConc(0,i1,i2,i3) = Conc(0,i1,i2,i3);
                //tempConc(1,i1,i2,i3) = Conc(1,i1,i2,i3);
            //}
        //}
    //}

#pragma omp parallel for collapse(2)
    for (int i1 = 0; i1 < L; i1++) {
        for (int i2 = 0; i2 < L; i2++) {
#pragma simd
            for (int i3 = 0; i3 < L; i3++) {
                int xUp = (i1+1);
                int xDown = (i1-1);
                int yUp = (i2+1);
                int yDown = (i2-1);
                int zUp = (i3+1);
                int zDown = (i3-1);

                for (int subInd = 0; subInd < 2; subInd++) {
			pong(subInd,i1,i2,i3) = ping(subInd,i1,i2,i3);
                    if (xUp<L) {
                        pong(subInd,i1,i2,i3) += (ping(subInd,xUp,i2,i3)-ping(subInd,i1,i2,i3))*D/6;
                    }
                    if (xDown>=0) {
                        pong(subInd,i1,i2,i3) += (ping(subInd,xDown,i2,i3)-ping(subInd,i1,i2,i3))*D/6;
                    }
                    if (yUp<L) {
                        pong(subInd,i1,i2,i3) += (ping(subInd,i1,yUp,i3)-ping(subInd,i1,i2,i3))*D/6;
                    }
                    if (yDown>=0) {
                        pong(subInd,i1,i2,i3) += (ping(subInd,i1,yDown,i3)-ping(subInd,i1,i2,i3))*D/6;
                    }
                    if (zUp<L) {
                        pong(subInd,i1,i2,i3) += (ping(subInd,i1,i2,zUp)-ping(subInd,i1,i2,i3))*D/6;
                    }
                    if (zDown>=0) {
                        pong(subInd,i1,i2,i3) += (ping(subInd,i1,i2,zDown)-ping(subInd,i1,i2,i3))*D/6;
                    }
                }
            }
        }
    }
    runDiffusionStep_sw.mark();
}

static void runDecayStep(float* Conc, int L, float mu) {
    runDecayStep_sw.reset();
    // computes the changes in substance concentrations due to decay
#pragma omp parallel for collapse(2)
    for (int i1 = 0; i1 < L; i1++) {
        for (int i2 = 0; i2 < L; i2++) {
#pragma simd
            for (int i3 = 0; i3 < L; i3++) {
                Conc(0,i1,i2,i3) = Conc(0,i1,i2,i3)*(1-mu);
                Conc(1,i1,i2,i3) = Conc(1,i1,i2,i3)*(1-mu);
            }
        }
    }
    runDecayStep_sw.mark();
}

static int cellMovementAndDuplication(float** posAll, float* pathTraveled, int* typesAll, int* numberDivisions, float pathThreshold, int divThreshold, int n) {
    cellMovementAndDuplication_sw.reset();
    int c;
    int currentNumberCells = n;
    float currentNorm;

    float currentCellMovement[3];
    float duplicatedCellOffset[3];


    for (c=0; c<n; c++) {
        // random cell movement
        currentCellMovement[0]=RandomFloatPos()-0.5;
        currentCellMovement[1]=RandomFloatPos()-0.5;
        currentCellMovement[2]=RandomFloatPos()-0.5;
        currentNorm = getNorm(currentCellMovement);
        posAll[c][0]+=0.1*currentCellMovement[0]/currentNorm;
        posAll[c][1]+=0.1*currentCellMovement[1]/currentNorm;
        posAll[c][2]+=0.1*currentCellMovement[2]/currentNorm;
        pathTraveled[c]+=0.1;

        // cell duplication if conditions fulfilled
        if (numberDivisions[c]<divThreshold) {

            if (pathTraveled[c]>pathThreshold) {
                pathTraveled[c]-=pathThreshold;
                numberDivisions[c]+=1;  // update number of divisions this cell has undergone
                currentNumberCells++;   // update number of cells in the simulation

                numberDivisions[currentNumberCells-1]=numberDivisions[c];   // update number of divisions the duplicated cell has undergone
                typesAll[currentNumberCells-1]=-typesAll[c]; // assign type of duplicated cell (opposite to current cell)

                // assign location of duplicated cell
                duplicatedCellOffset[0]=RandomFloatPos()-0.5;
                duplicatedCellOffset[1]=RandomFloatPos()-0.5;
                duplicatedCellOffset[2]=RandomFloatPos()-0.5;
                currentNorm = getNorm(duplicatedCellOffset);
                posAll[currentNumberCells-1][0]=posAll[c][0]+0.05*duplicatedCellOffset[0]/currentNorm;
                posAll[currentNumberCells-1][1]=posAll[c][1]+0.05*duplicatedCellOffset[1]/currentNorm;
                posAll[currentNumberCells-1][2]=posAll[c][2]+0.05*duplicatedCellOffset[2]/currentNorm;

            }

        }
    }
    cellMovementAndDuplication_sw.mark();
    return currentNumberCells;
}

static void runDiffusionClusterStep(float* Conc, float** movVec, float** posAll, int* typesAll, int n, int L, float speed) {
    runDiffusionClusterStep_sw.reset();
    // computes movements of all cells based on gradients of the two substances

    float sideLength = 1/(float)L; // length of a side of a diffusion voxel


#pragma omp parallel for
    for (int c = 0; c < n; c++) {
	    float gradSub1[3];
	    float gradSub2[3];

        int i1 = std::min((int)floor(posAll[c][0]/sideLength),(L-1));
        int i2 = std::min((int)floor(posAll[c][1]/sideLength),(L-1));
        int i3 = std::min((int)floor(posAll[c][2]/sideLength),(L-1));

        int xUp = std::min((i1+1),L-1);
        int xDown = std::max((i1-1),0);
        int yUp = std::min((i2+1),L-1);
        int yDown = std::max((i2-1),0);
        int zUp = std::min((i3+1),L-1);
        int zDown = std::max((i3-1),0);

        gradSub1[0] = (Conc(0,xUp,i2,i3)-Conc(0,xDown,i2,i3))/(sideLength*(xUp-xDown));
        gradSub1[1] = (Conc(0,i1,yUp,i3)-Conc(0,i1,yDown,i3))/(sideLength*(yUp-yDown));
        gradSub1[2] = (Conc(0,i1,i2,zUp)-Conc(0,i1,i2,zDown))/(sideLength*(zUp-zDown));

        gradSub2[0] = (Conc(1,xUp,i2,i3)-Conc(1,xDown,i2,i3))/(sideLength*(xUp-xDown));
        gradSub2[1] = (Conc(1,i1,yUp,i3)-Conc(1,i1,yDown,i3))/(sideLength*(yUp-yDown));
        gradSub2[2] = (Conc(1,i1,i2,zUp)-Conc(1,i1,i2,zDown))/(sideLength*(zUp-zDown));

        float normGrad1 = getNorm(gradSub1);
        float normGrad2 = getNorm(gradSub2);

        if ((normGrad1>0)&&(normGrad2>0)) {
            movVec[c][0]=typesAll[c]*(gradSub1[0]/normGrad1-gradSub2[0]/normGrad2)*speed;
            movVec[c][1]=typesAll[c]*(gradSub1[1]/normGrad1-gradSub2[1]/normGrad2)*speed;
            movVec[c][2]=typesAll[c]*(gradSub1[2]/normGrad1-gradSub2[2]/normGrad2)*speed;
        }

        else {
            movVec[c][0]=0;
            movVec[c][1]=0;
            movVec[c][2]=0;
        }
    }
    runDiffusionClusterStep_sw.mark();
}

static void getParameters(int targetN, int n, float** posAll, int& nrCellsSubVol, float**& posSubvol, int* typesSubvol, int* typesAll){
	extra_sw.reset();
    float subVolMax = pow(float(targetN)/float(n),1.0/3.0)/2;

    // the locations of all cells within the subvolume are copied to array posSubvol
    for (int i1 = 0; i1 < n; i1++) {
        posSubvol[i1] = new float[3];
        if ((fabs(posAll[i1][0]-0.5)<subVolMax) && (fabs(posAll[i1][1]-0.5)<subVolMax) && (fabs(posAll[i1][2]-0.5)<subVolMax)) {
            posSubvol[nrCellsSubVol][0] = posAll[i1][0];
            posSubvol[nrCellsSubVol][1] = posAll[i1][1];
            posSubvol[nrCellsSubVol][2] = posAll[i1][2];
            typesSubvol[nrCellsSubVol] = typesAll[i1];

            nrCellsSubVol++;
        }
    }

    if(quiet < 1)
        printf("number of cells in subvolume: %d\n", nrCellsSubVol);


    extra_sw.mark();
}

static float getEnergy(float** posAll, int* typesAll, int n, float spatialRange, int targetN, int nrCellsSubVol, float** posSubvol, int* typesSubvol) {
    getEnergy_sw.reset();
    // Computes an energy measure of clusteredness within a subvolume. The size of the subvolume
    // is computed by assuming roughly uniform distribution within the whole volume, and selecting
    // a volume comprising approximately targetN cells.
    float intraClusterEnergy = 0.0;
    float extraClusterEnergy = 0.0;
    float nrSmallDist=0.0;
    float intraClusterEnergy_array[nrCellsSubVol][nrCellsSubVol];
    float extraClusterEnergy_array[nrCellsSubVol][nrCellsSubVol];
    float nrSmallDist_array[nrCellsSubVol][nrCellsSubVol];

#pragma omp parallel for
    for (int i1 = 0; i1 < nrCellsSubVol; i1++) {
        for (int i2 = i1+1; i2 < nrCellsSubVol; i2++) {
            float currDist =  getL2Distance(posSubvol[i1][0],posSubvol[i1][1],posSubvol[i1][2],posSubvol[i2][0],posSubvol[i2][1],posSubvol[i2][2]);
            if (currDist<spatialRange) {
                nrSmallDist_array[i1][i2] = 1;//currDist/spatialRange;
                if (typesSubvol[i1]*typesSubvol[i2]>0) {
                    intraClusterEnergy_array[i1][i2] = fmin(100.0,spatialRange/currDist); 
                    extraClusterEnergy_array[i1][i2] = 0;
		} else {
                    extraClusterEnergy_array[i1][i2] = fmin(100.0,spatialRange/currDist);
                    intraClusterEnergy_array[i1][i2] = 0;
                }
            } else {
		    intraClusterEnergy_array[i1][i2] = 0;
		    extraClusterEnergy_array[i1][i2] = 0;
		    nrSmallDist_array[i1][i2] = 0;
	    }
        }
    }
#pragma omp parallel for reduction( + : nrSmallDist, extraClusterEnergy, intraClusterEnergy)
    for( int i1 = 0; i1 < nrCellsSubVol; i1++ ){
	    for( int i2 = i1+1; i2 < nrCellsSubVol; i2++ ){
		    nrSmallDist += nrSmallDist_array[i1][i2];
		    extraClusterEnergy += extraClusterEnergy_array[i1][i2];
		    intraClusterEnergy += intraClusterEnergy_array[i1][i2];
	    }
    }
    float totalEnergy = (extraClusterEnergy-intraClusterEnergy)/(1.0+100.0*nrSmallDist);
    getEnergy_sw.mark();
    return totalEnergy;
}

static bool getCriterion(float** posAll, int* typesAll, int n, float spatialRange, int targetN, int nrCellsSubVol, float** posSubvol, int* typesSubvol) {
    getCriterion_sw.reset();
    // Returns 0 if the cell locations within a subvolume of the total system, comprising approximately targetN cells,
    // are arranged as clusters, and 1 otherwise.

    int i1, i2;
    int nrClose=0;      // number of cells that are close (i.e. within a distance of spatialRange)
    float currDist;
    int sameTypeClose=0; // number of cells of the same type, and that are close (i.e. within a distance of spatialRange)
    int diffTypeClose=0; // number of cells of opposite types, and that are close (i.e. within a distance of spatialRange)


    // If there are not enough cells within the subvolume, the correctness criterion is not fulfilled
    if ((((float)(nrCellsSubVol))/(float)targetN) < 0.25) {
        getCriterion_sw.mark();
        if(quiet < 2)
            printf("not enough cells in subvolume: %d\n", nrCellsSubVol);
        return false;
    }

    // If there are too many cells within the subvolume, the correctness criterion is not fulfilled
    if ((((float)(nrCellsSubVol))/(float)targetN) > 4) {
        getCriterion_sw.mark();
        if(quiet < 2)
            printf("too many cells in subvolume: %d\n", nrCellsSubVol);
        return false;
    }

    for (i1 = 0; i1 < nrCellsSubVol; i1++) {
        for (i2 = i1+1; i2 < nrCellsSubVol; i2++) {
            currDist =  getL2Distance(posSubvol[i1][0],posSubvol[i1][1],posSubvol[i1][2],posSubvol[i2][0],posSubvol[i2][1],posSubvol[i2][2]);
            if (currDist<spatialRange) {
                nrClose++;
                if (typesSubvol[i1]*typesSubvol[i2]<0) {
                    diffTypeClose++;
                }
                else {
                    sameTypeClose++;
                }
            }
        }
    }

    float correctness_coefficient = ((float)diffTypeClose)/(nrClose+1.0);

    // check if there are many cells of opposite types located within a close distance, indicative of bad clustering
    if (correctness_coefficient > 0.1) {
        getCriterion_sw.mark();
        if(quiet < 2)
            printf("cells in subvolume are not well-clustered: %f\n", correctness_coefficient);
        return false;
    }

    // check if clusters are large enough, i.e. whether cells have more than 100 cells of the same type located nearby
    float avgNeighbors = ((float)sameTypeClose/nrCellsSubVol);
    if(quiet < 1)
        printf("average neighbors in subvolume: %f\n", avgNeighbors);
    if (avgNeighbors < 100) {
        getCriterion_sw.mark();
        if(quiet < 2)
            printf("cells in subvolume do not have enough neighbors: %f\n", avgNeighbors);
        return false;
    }


    if(quiet < 1)
        printf("correctness coefficient: %f\n", correctness_coefficient);

    getCriterion_sw.mark();
    return true;
}

static const char usage_str[] = "USAGE:\t%s[-h] [-V] [--<param>=<value>]* <input file> \n";

static void usage(const char *name)
{
    die(usage_str, basename(name));
}

static void help(const char *name)
{
    fprintf(stderr, usage_str, name);
    fprintf(stderr, "DESCRIPTION\n"
            "\t Clustering of Cells in 3D space by movements along substance gradients\n"
            "\t In this simulation, there are two phases. In a first phase, a\n"
            "\t single initial cell moves randomly in 3 dimensional space and\n"
            "\t recursively gives rise to daughter cell by duplication. In the\n"
            "\t second phase, cells move along the gradients of their preferred\n"
            "\t substance. There are two substances in this example, and cells\n"
            "\t produce the same substance as they prefer. The substances\n"
            "\t diffuses and decays in 3D space.\n");
    fprintf(stderr, "PARAMETERS\n"
            "\t <input file> should have <param>=<value> for each of the following:\n"
            "\t speed\n\t    multiplicative factor for speed of gradient-based movement of the cells (float)\n"
            "\t T\n\t    Number of time steps of simulated cell movements (int64_t)\n"
            "\t L\n\t    Defines resolution of diffusion mesh (int64_t)\n"
            "\t D\n\t    Diffusion constant (float)\n"
            "\t mu\n\t    Decay constant (float)\n"
            "\t divThreshold\n\t    number of divisions a cell can maximally undergo (relevant only for the first phase of the simulation) (unsigned)\n"
            "\t finalNumberCells\n\t    Number of cells after cells have recursively duplicated (divided) (int64_t)\n"
            "\t spatialRange\n\t    defines the maximal spatial extend of the clusters. This parameter is only used for computing the energy function and the correctness criterion (float)\n");
    fprintf(stderr, "OPTIONS\n"
            "\t-h,--help\n\t    print this help message\n"
            "\t-v,--version\n\t    print configuration information\n"
            "\t-q,--quiet\n\t    lower output to stdout. Multiples accepted.\n"
            "\t-v,--verbose\n\t    increase output to stdout. Multiples accepted\n"
            "\t--<param>=<value>\n\t    override param/value form input file\n");
}

int main(int argc, char *argv[]) {
    stopwatch init_sw;
    init_sw.reset();

    const option opts[] =
    {
        {"help",            no_argument,       0, 'h'},
        {"version",         no_argument,       0, 'V'},
        {"quiet",           no_argument,       0, 'q'},
        {"verbose",         no_argument,       0, 'v'},
        {0, 0, 0, 0},
    };

    vector<char*> candidate_kvs;

    int opt;
    do
    {
        int in_ind = optind;
        opterr     = 0;
        opt        = getopt_long(argc, argv, "hVqv", opts, 0);
        switch(opt)
        {
        case 0:
            break;
        case '?':
            if(optopt == 0)
            {
                candidate_kvs.push_back(read_kv(argv, in_ind, &optind));
            }
            break;
        case 'h':
            help(argv[0]);
            exit(0);
        case 'V':
            print_sys_config(stderr);
            exit(0);
        case 'q':
            ++quiet;
            break;
        case 'v':
            --quiet;
            break;
        default:
            usage(argv[0]);
        case -1:
            break;
        };
    }
    while(opt != -1);

    if(optind+1 < argc)
        usage(argv[0]);

    fprintf(stderr, "==================================================\n");
    fprintf(stderr, "NAME                                = parallel_red_getenergy\n"); // title

    print_sys_config(stderr);

    const cdc_params params = get_params(argv[optind], candidate_kvs, quiet);

    print_params(&params, stderr);

    const float    speed            = params.speed;
    const int64_t  T                = params.T;
    const int64_t  L                = params.L;
    const float    D                = params.D;
    const float    mu               = params.mu;
    const unsigned divThreshold     = params.divThreshold;
    const int64_t  finalNumberCells = params.finalNumberCells;
    const float    spatialRange     = params.spatialRange;
    const float    pathThreshold    = params.pathThreshold;

    int i,c,d;
    int i1, i2, i3, i4;

    float energy;   // value that quantifies the quality of the cell clustering output. The smaller this value, the better the clustering.

    float** posAll=0;   // array of all 3 dimensional cell positions
    posAll = new float*[finalNumberCells];
    float** currMov=0;  // array of all 3 dimensional cell movements at the last time point
    currMov = new float*[finalNumberCells]; // array of all cell movements in the last time step
    float zeroFloat = 0.0;

    float pathTraveled[finalNumberCells];   // array keeping track of length of path traveled until cell divides
    int numberDivisions[finalNumberCells];  //array keeping track of number of division a cell has undergone
    int typesAll[finalNumberCells];     // array specifying cell type (+1 or -1)

    numberDivisions[0]=0;   // the first cell has initially undergone 0 duplications (= divisions)
    typesAll[0]=1;  // the first cell is of type 1

    bool currCriterion;

    // Initialization of the various arrays
    for (i1 = 0; i1 < finalNumberCells; i1++) {
        currMov[i1] = new float[3];
        posAll[i1] = new float[3];
        pathTraveled[i1] = zeroFloat;
        pathTraveled[i1] = 0;
        for (i2 = 0; i2 < 3; i2++) {
            currMov[i1][i2] = zeroFloat;
            posAll[i1][i2] = 0.5;
        }
    }

    	Conc_Pad = L+(L%16 == 0? 0:16-L%16);
	float* Conc  = (float*) _mm_malloc( 2*Conc_Pad*Conc_Pad*Conc_Pad*sizeof(float), 16);
	float* Conc2 = (float*) _mm_malloc( 2*Conc_Pad*Conc_Pad*Conc_Pad*sizeof(float), 16);
	memset(Conc, 0, 2*Conc_Pad*Conc_Pad*Conc_Pad*sizeof(float) );
	memset(Conc2, 0, 2*Conc_Pad*Conc_Pad*Conc_Pad*sizeof(float) );

    init_sw.mark();
    fprintf(stderr, "%-35s = %le s\n",  "INITIALIZATION_TIME", init_sw.elapsed);

    stopwatch compute_sw;
    compute_sw.reset();

    stopwatch phase1_sw;
    phase1_sw.reset();

    int64_t n = 1; // initially, there is one single cell

    // Phase 1: Cells move randomly and divide until final number of cells is reached
    while (n<finalNumberCells) {
        produceSubstances(Conc, posAll, typesAll, L, n); // Cells produce substances. Depending on the cell type, one of the two substances is produced.
        runDiffusionStep(Conc, Conc2, L, D); // Simulation of substance diffusion
        runDecayStep(Conc2, L, mu);
        n = cellMovementAndDuplication(posAll, pathTraveled, typesAll, numberDivisions, pathThreshold, divThreshold, n);
	std::swap(Conc,Conc2);

        for (c=0; c<n; c++) {
            // boundary conditions
            for (d=0; d<3; d++) {
                if (posAll[c][d]<0) {posAll[c][d]=0;}
                if (posAll[c][d]>1) {posAll[c][d]=1;}
            }
        }
    }
    phase1_sw.mark();
    fprintf(stderr, "%-35s = %le s\n",  "PHASE1_TIME", phase1_sw.elapsed);

    stopwatch phase2_sw;
    phase2_sw.reset();

    // Phase 2: Cells move along the substance gradients and cluster
    for (i=0; i<T; i++) {

        if ((i%10) == 0) {
            if(quiet < 1) {
                printf("step %d\n", i);
            }
            else if(quiet < 2) {
                printf("\rstep %d", i);
                fflush(stdout);
            }
        }

        if(quiet == 1)
            printf("\n");

        if (i==0) {
		float** posSubvol=0;
		posSubvol = new float*[n];
		int typesSubvol[n];
		int nrCellsSubVol = 0;

		getParameters(10000, n, posAll, nrCellsSubVol, posSubvol, typesSubvol, typesAll);
            energy = getEnergy(posAll, typesAll, n, spatialRange, 10000, nrCellsSubVol, posSubvol, typesSubvol);
            currCriterion = getCriterion(posAll, typesAll, n, spatialRange, 10000, nrCellsSubVol, posSubvol, typesSubvol);
            fprintf(stderr, "%-35s = %d\n",  "INITIAL_CRITERION", currCriterion);
            fprintf(stderr, "%-35s = %le\n", "INITIAL_ENERGY", energy);
        }

        if (i==(T-1)) {
		float** posSubvol=0;
		posSubvol = new float*[n];
		int typesSubvol[n];
		int nrCellsSubVol = 0;

		getParameters(10000, n, posAll, nrCellsSubVol, posSubvol, typesSubvol, typesAll);
            energy = getEnergy(posAll, typesAll, n, spatialRange, 10000, nrCellsSubVol, posSubvol, typesSubvol);
            currCriterion = getCriterion(posAll, typesAll, n, spatialRange, 10000, nrCellsSubVol, posSubvol, typesSubvol);
            fprintf(stderr, "%-35s = %d\n",  "FINAL_CRITERION", currCriterion);
            fprintf(stderr, "%-35s = %le\n", "FINAL_ENERGY", energy);

        }

        produceSubstances(Conc, posAll, typesAll, L, n);
        runDiffusionStep(Conc, Conc2, L, D);
        runDecayStep(Conc2, L, mu);
        runDiffusionClusterStep(Conc2, currMov, posAll, typesAll, n, L, speed);
		std::swap(Conc,Conc2);

        for (c=0; c<n; c++) {
            posAll[c][0] = posAll[c][0]+currMov[c][0];
            posAll[c][1] = posAll[c][1]+currMov[c][1];
            posAll[c][2] = posAll[c][2]+currMov[c][2];

            // boundary conditions: cells can not move out of the cube [0,1]^3
            for (d=0; d<3; d++) {
                if (posAll[c][d]<0) {posAll[c][d]=0;}
                if (posAll[c][d]>1) {posAll[c][d]=1;}
            }
        }
    }
    phase2_sw.mark();
    compute_sw.mark();
    fprintf(stderr, "%-35s = %le s\n",  "PHASE2_TIME", phase2_sw.elapsed);


    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "produceSubstances_TIME",          produceSubstances_sw.elapsed, produceSubstances_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "runDiffusionStep_TIME",           runDiffusionStep_sw.elapsed, runDiffusionStep_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "runDecayStep_TIME",               runDecayStep_sw.elapsed, runDecayStep_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "cellMovementAndDuplication_TIME", cellMovementAndDuplication_sw.elapsed, cellMovementAndDuplication_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "runDiffusionClusterStep_TIME",    runDiffusionClusterStep_sw.elapsed, runDiffusionClusterStep_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "getEnergy_TIME",                  getEnergy_sw.elapsed, getEnergy_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "getCriterion_TIME",               getCriterion_sw.elapsed, getCriterion_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "extra_TIME",                      extra_sw.elapsed, extra_sw.elapsed*100.0f/compute_sw.elapsed);
    fprintf(stderr, "%-35s = %le s (%3.2f %%)\n", "TOTAL_COMPUTE_TIME",              compute_sw.elapsed, compute_sw.elapsed*100.0f/compute_sw.elapsed);

    fprintf(stderr, "==================================================\n");

    return 0;
}
