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
	SR_RESAMPLE_NEIGHBOR = 12,
	SR_CHECK_VISIBILITY = 13
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
	float incidentAngle; // the incident angle coming out of the reconnection vertex in the base path
	uint pathLength; // the length of the path at the reconnection vertex
	Spectrum postfixRadiance[FILM_MAX_RADIANCE_GROUP_COUNT]; // the radiance of the path of the path at the reconnection vertex and after
	Point hitPoint; // contains info on the exact hit point on the reconnection vertex
	// TODO: find out if LuxCoreRender has multi-lobed materials
	// uint prevLobeIndex; // the sampled lobe index of the material at the previous vertex 
	// uint currLobeIndex; // the sampled lobe index of the material at the reconnection vertex
} ReconnectionVertex;

// Stores reuse information about a selected ReSPIR sample. (spatial reuse only)
typedef struct {
	ReconnectionVertex reconnectionVertex; // the chosen reconnection vertex for this path
	Spectrum prefixRadiance[FILM_MAX_RADIANCE_GROUP_COUNT]; // the radiance of the path at the vertices before the reconnection vertex
	SampleResult sampleResult; // the cached sampleresult data of the entire path
	BSDF prefixBsdf; // the BSDF point where the vertex before the reconnection vertex was hit
	float hitTime; // time the reconnection vertex was hit. we use this to shoot a visibility ray backwards to the connecting offset path vertex
	Seed seedInitial; // the initial GPUTask seed at the beginning of tracing this path
	Seed seedReconnectionVertex; // the GPUTask seed right after tracing the reconnection vertex
	uint pathLength; // the length of the path
	float partialCachedJacobian; // the denominator of the jacobian for this path for calculating the full jacobian when performing reuse 
} RespirSample;

// A streaming random-sampling reservoir for spatial reuse.
typedef struct {
	RespirSample selectedSample; // selected sample result
	float selectedWeight; // (unbiased contribution) weight of selected sample
	float sumWeight; // sum weights
} RespirReservoir;

// The state used to keep track of the rendered path
typedef struct {
	PathState state;

	Spectrum throughput;
	BSDF bsdf; // Variable size structure

	Seed seedPassThroughEvent;
	Seed seedReservoirSampling;

	// keep track of the MIS weights of the most recent direct lighting event
	Spectrum lastWeight;
	// product of all bsdfPdfW
	float bsdfPdfWProduct;
	
	// TODO: MOVE INTO SEPARATE BUFFER IN THE FUTURE
	uint timeBeforeSpatialReuse; // save time before spatial reuse to make sure rays after spatial reuse are using the correct time
	// Reservoir data structure for initial path resampling using RIS
	RespirReservoir initialPathReservoir;
	
	// Neighbor search info
	int currentNeighborGid, neighborSearchDx, neighborSearchDy;
	
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

	// Swap reservoir to use to hold the last spatial passes's reservoirs in order to prevent race conditions.
	RespirReservoir tmpReservoir;
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
