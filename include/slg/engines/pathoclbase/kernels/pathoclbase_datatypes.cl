#line 2 "pathoclbase_datatypes.cl"

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

//------------------------------------------------------------------------------
// GPUTask data types
//------------------------------------------------------------------------------

typedef enum {
	// Micro-kernel states
	MK_RT_NEXT_VERTEX = 0,
	MK_HIT_NOTHING = 1,
	MK_HIT_OBJECT = 2,
	MK_DL_ILLUMINATE = 3,
	MK_DL_SAMPLE_BSDF = 4,
	MK_RT_DL = 5,
	MK_GENERATE_NEXT_VERTEX_RAY = 6,
	MK_SPLAT_SAMPLE = 7,
	MK_NEXT_SAMPLE = 8,
	MK_GENERATE_CAMERA_RAY = 9,
	MK_DONE = 10,
	SYNC = 11,
	SR_MK_NEXT_NEIGHBOR = 12,
	SR_MK_SHIFT = 13,
	SR_MK_CHECK_VISIBILITY = 14,
	SR_MK_RESAMPLE = 15,
	SR_MK_FINISH_RESAMPLE = 16

} PathState;

typedef struct {
	union {
		struct {
			// Must be a power of 2
			unsigned int previewResolutionReduction, previewResolutionReductionStep;
			unsigned int resolutionReduction;
		} rtpathocl;
	} renderEngine;

	Scene scene;
	Sampler sampler;
	PathTracer pathTracer;
	Filter pixelFilter;
	Film film;
} GPUTaskConfiguration;

typedef struct {
	unsigned int lightIndex;
	float pickPdf;

	float directPdfW;

	// Radiance to add to the result if light source is visible
	// Note: it doesn't include the pathThroughput
	Spectrum lightRadiance;
	// This is used only if Film channel IRRADIANCE is enabled and
	// only for the first path vertex
	Spectrum lightIrradiance;

	unsigned int lightID;
} DirectLightIlluminateInfo;

// Stores information about the reconnection vertex for a particular path in the ReSTIR algorithm.
typedef struct {
	Spectrum irradiance[FILM_MAX_RADIANCE_GROUP_COUNT]; // the radiance of the path of the path at the rc vertex and after
	BSDF bsdf; // contains info on the exact hit point on the rc vertex
	// the bsdf contains the incident direction coming into the reconnection vertex
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
} RespirSample;

// A streaming random-sampling reservoir for spatial reuse.
typedef struct {
	RespirSample sample; // selected sample result
	float M; // sample count
	float weight; // sum weight when used during initial path resampling 
				  // otherwise (unbiased contribution) weight of selected sample 1/p(sample) * 1/M * w_sum
} RespirReservoir;

// The state used to keep track of the rendered path
typedef struct {
	// TODO: MOVE PATHSTATE INTO SEPARATE STRUCTURE IN THE FUTURE
	PathState state;

	Spectrum throughput;
	BSDF bsdf; // Variable size structure

	Seed seedPassThroughEvent;
	Seed seedReservoirSampling;

	// keep track of cumulative products
	float pathPdf; // bsdfProduct and connectionThroughput
	float rrProbProd;
	float lastDirectLightPdf; // for direct light illumination sampled from NEE (+ cheater BSDF)

	// TODO: MOVE INTO SEPARATE BUFFER IN THE FUTURE
	uint preSpatialReuseTime; // save time before spatial reuse to make sure rays after spatial reuse are using the correct time
	// Reservoir data structure for initial path resampling using RIS
	RespirReservoir reservoir;
	
	// Neighbor search info
	int neighborGid;
	uint numNeighborsLeft;
	uint numValidNeighbors;

	// For each reuse
	float canonicalMisWeight;
	RespirReservoir spatialReuseReservoir;

	// MK_SHIFT kernel inputs and outputs
	// Shift from src reservoir to dst reservoir and store results in shiftReservoir
	// Executes the kernel stored in afterShiftState afterwards.
	// TODO: only need this for the ASYNC kernels, move out of TaskState
	uint shiftSrcGid, shiftDstGid;
	RespirReservoir shiftReservoir;
	PathState afterShiftState;
	
	int albedoToDo, photonGICacheEnabledOnLastHit,
			photonGICausticCacheUsed, photonGIShowIndirectPathMixUsed,
			// The shadow transparency lag used by Scene_Intersect()
			throughShadowTransparency;
} RespirGPUTaskState;

typedef struct {
	PathState state;

	Spectrum throughput;
	BSDF bsdf; // Variable size structure

	Seed seedPassThroughEvent;
	
	int albedoToDo, photonGICacheEnabledOnLastHit,
			photonGICausticCacheUsed, photonGIShowIndirectPathMixUsed,
			// The shadow transparency lag used by Scene_Intersect()
			throughShadowTransparency;
} VanillaGPUTaskState;

typedef enum {
	ILLUMINATED, SHADOWED, NOT_VISIBLE
} DirectLightResult;

typedef struct {
	// Used to store some intermediate result
	DirectLightIlluminateInfo illumInfo;

	Seed seedPassThroughEvent;

	DirectLightResult directLightResult;

	// The shadow transparency flag used by Scene_Intersect()
	int throughShadowTransparency;
} GPUTaskDirectLight;

typedef struct {
	// The task seed
	Seed seed;

	// Space for temporary storage
	BSDF tmpBsdf; // Variable size structure

	// This is used by TriangleLight_Illuminate() to temporary store the
	// point on the light sources.
	// Also used by Scene_Intersect() for evaluating volume textures.
	HitPoint tmpHitPoint;
	
	// This is used by DirectLight_BSDFSampling()
	PathDepthInfo tmpPathDepthInfo;
} VanillaGPUTask;

typedef struct {
	// The task seed
	Seed seed;

	// Space for temporary storage
	BSDF tmpBsdf; // Variable size structure

	// This is used by TriangleLight_Illuminate() to temporary store the
	// point on the light sources.
	// Also used by Scene_Intersect() for evaluating volume textures.
	HitPoint tmpHitPoint;
	
	// This is used by DirectLight_BSDFSampling()
	PathDepthInfo tmpPathDepthInfo;
} RespirGPUTask;


#if defined(RENDER_ENGINE_RESPIRPATHOCL) 
typedef RespirGPUTaskState GPUTaskState;
typedef RespirGPUTask GPUTask;
#else
typedef VanillaGPUTaskState GPUTaskState;
typedef VanillaGPUTask GPUTask;
#endif


typedef struct {
	unsigned int sampleCount;
} GPUTaskStats;
