#line 2 "respir_types.cl"

/***************************************************************************
 * Copyright 1998-2020 by authors (see AUTHORS.txt)                        *
 *                                                                         *
 *   This file is part of LuxCoreRender.                                   *
 *                                                                         *
 * Licensed under the Apache License, Version 2.0 (the "License");         *
 * you may not use this file except in compliance with the License.        *
 * You may obtain a copy of the License at                                 *
 *                                                                         *
 *     http://www.apache.org/licenses/LICENSE-2.0                          *
 *                                                                         *
 * Unless required by applicable law or agreed to in writing, software     *
 * distributed under the License is distributed on an "AS IS" BASIS,       *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.*
 * See the License for the specific language governing permissions and     *
 * limitations under the License.                                          *
 ***************************************************************************/

//------------------------------------------------------------------------------
// Some OpenCL specific definition
//------------------------------------------------------------------------------

#if defined(SLG_OPENCL_KERNEL)

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#endif

/*
// Respir types
*/

typedef enum {
	SYNC = 100,
	SR_MK_NEXT_NEIGHBOR = 101,
	SR_MK_SHIFT = 102,
	SR_MK_CHECK_VISIBILITY = 103,
	SR_MK_RESAMPLE = 104,
	SR_MK_FINISH_RESAMPLE = 105
} RespirAsyncState;

// Stores information about the reconnection vertex for a particular path in the ReSTIR algorithm.
typedef struct {
	Spectrum irradiance[FILM_MAX_RADIANCE_GROUP_COUNT]; // the radiance of the path of the path at the rc vertex and after
	BSDF bsdf; // contains info on the exact hit point on the rc vertex
	Vector incidentDir; // cache the scatter direction from the next vertex to the rc vertex
	float incidentPdf; // cache pdf from rc vertex towards incident dir
	Spectrum incidentBsdfValue; // cache bsdf value from rc vertex towards incident dir
	float prefixToRcPdf; // cache pdf from prefix point towards this rc vertex 
	float jacobian; // cache the prefix vertex part of the jacobian (squared distance to self rc vertex / cos angle to norm of rc vertex)
	int pathDepth; // -1 (no rcVertex), 0 is primary hit, 1 is secondary hit, ...
} RcVertex;

// Stores reuse information about a selected ReSPIR sample. (spatial reuse only)
typedef struct {
	Spectrum integrand[FILM_MAX_RADIANCE_GROUP_COUNT]; // the cached integrand of the path, updated each time a new sample is selected
	BSDF prefixBsdf; // the BSDF point where the vertex before the rc vertex was hit
	RcVertex rc; // the chosen rc vertex for this path
	float lightPdf; // the NEE light pick probability of the sample
	float hitTime; // time the prefix vertex was hit. we use this to shoot a visibility ray to the rc vertex
	int pathDepth;
} RespirSample;

// A streaming random-sampling reservoir for spatial reuse.
typedef struct {
	RespirSample sample; // selected sample result
	float M; // sample count
	float weight; // sum weight when used during initial path resampling 
				  // otherwise (unbiased contribution) weight of selected sample 1/p(sample) * 1/M * w_sum
} RespirReservoir;

// The state used to keep track of the rendered path
// And initial path resampling
typedef struct {
	Spectrum throughput;
	BSDF bsdf; // Variable size structure

	Seed seedPassThroughEvent;
	Seed seedReservoirSampling;

    // keep track of cumulative products
	Spectrum pathPdf; // bsdfProduct and connectionThroughput
	float rrProbProd; // running product of russian roulette probability each vertex hit
	float lastDirectLightPdf; // for direct light illumination sampled from NEE (+ cheater BSDF)
	
	// initial path resampling reservoir
	RespirReservoir reservoir;

	int albedoToDo, photonGICacheEnabledOnLastHit,
			photonGICausticCacheUsed, photonGIShowIndirectPathMixUsed,
			// The shadow transparency lag used by Scene_Intersect()
			throughShadowTransparency;
} RespirGPUTaskState;

typedef RespirGPUTaskState GPUTaskState;

// Data only used for spatial reuse
typedef struct {
	uint preSpatialReuseTime; // save time before spatial reuse to make sure rays after spatial reuse are using the correct time
	
	// Neighbor search info
	int neighborGid;
	uint numNeighborsLeft;
	uint numValidNeighbors;

	// For each iteration
	float canonicalMisWeight;

    // Spatial reuse GRIS accumulated in here
	RespirReservoir spatialReuseReservoir;
} SpatialReuseData;

// MK_SHIFT kernel inputs and outputs
typedef struct {
	// Shift from src reservoir to dst reservoir and store results in shiftReservoir
	uint shiftSrcGid, shiftDstGid;
	RespirReservoir shiftReservoir;
    // Executes the kernel stored in afterShiftState afterwards.
	RespirAsyncState afterShiftState;
} ShiftInOutData;


/*
// Kernel arg macros
*/

// Reservoir data structure for initial path resampling using RIS
#define KERNEL_ARGS_SPATIALREUSE \
		, __global int* pixelIndexMap \
        , __global SpatialReuseData* spatialReuseDatas \
		, const uint spatialRadius \
		, const uint numSpatialNeighbors

#define KERNEL_ARGS_SHIFT \
        , __global ShiftInOutData* shiftInOutDatas