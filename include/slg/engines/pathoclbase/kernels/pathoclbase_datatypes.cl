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
	MK_DONE = 10
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

typedef struct {
	Spectrum throughput;
	BSDF bsdf; // Variable size structure

	Seed seedPassThroughEvent;
	
	int albedoToDo, photonGICacheEnabledOnLastHit,
			photonGICausticCacheUsed, photonGIShowIndirectPathMixUsed,
			// The shadow transparency lag used by Scene_Intersect()
			throughShadowTransparency;
} VanillaGPUTaskState;

#if !defined(RENDER_ENGINE_RESPIRPATHOCL)
typedef VanillaGPUTaskState GPUTaskState;
#endif

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
} GPUTask;

typedef struct {
	unsigned int sampleCount;
} GPUTaskStats;

//------------------------------------------------------------------------------
// Kernel parameters
//------------------------------------------------------------------------------

#define KERNEL_ARGS_VOLUMES \
		, __global PathVolumeInfo *directLightVolInfos

#define KERNEL_ARGS_INFINITELIGHTS \
		, const float worldCenterX \
		, const float worldCenterY \
		, const float worldCenterZ \
		, const float worldRadius

#define KERNEL_ARGS_NORMALS_BUFFER \
		, __global const Normal* restrict vertNormals
#define KERNEL_ARGS_TRINORMALS_BUFFER \
		, __global const Normal* restrict triNormals
#define KERNEL_ARGS_UVS_BUFFER \
		, __global const UV* restrict vertUVs
#define KERNEL_ARGS_COLS_BUFFER \
		, __global const Spectrum* restrict vertCols
#define KERNEL_ARGS_ALPHAS_BUFFER \
		, __global const float* restrict vertAlphas
#define KERNEL_ARGS_VERTEXAOVS_BUFFER \
		, __global const float* restrict vertexAOVs
#define KERNEL_ARGS_TRIAOVS_BUFFER \
		, __global const float* restrict triAOVs

#define KERNEL_ARGS_ENVLIGHTS \
		, __global const uint* restrict envLightIndices \
		, const uint envLightCount

#define KERNEL_ARGS_INFINITELIGHT \
		, __global const float* restrict envLightDistribution

#define KERNEL_ARGS_IMAGEMAPS_PAGES \
		, __global const ImageMap* restrict imageMapDescs \
		, __global const float* restrict imageMapBuff0 \
		, __global const float* restrict imageMapBuff1 \
		, __global const float* restrict imageMapBuff2 \
		, __global const float* restrict imageMapBuff3 \
		, __global const float* restrict imageMapBuff4 \
		, __global const float* restrict imageMapBuff5 \
		, __global const float* restrict imageMapBuff6 \
		, __global const float* restrict imageMapBuff7

#define KERNEL_ARGS_FAST_PIXEL_FILTER \
		, __global float *pixelFilterDistribution

#define KERNEL_ARGS_PHOTONGI \
		, __global const RadiancePhoton* restrict pgicRadiancePhotons \
		, uint pgicLightGroupCounts \
		, __global const Spectrum* restrict pgicRadiancePhotonsValues \
		, __global const IndexBVHArrayNode* restrict pgicRadiancePhotonsBVHNodes \
		, __global const Photon* restrict pgicCausticPhotons \
		, __global const IndexBVHArrayNode* restrict pgicCausticPhotonsBVHNodes

#define KERNEL_ARGS \
		__global PathState* pathStates \
		, __constant const GPUTaskConfiguration* restrict taskConfig \
		, __global GPUTask *tasks \
		, __global GPUTaskDirectLight *tasksDirectLight \
		, __global GPUTaskState *tasksState \
		, __global GPUTaskStats *taskStats \
		KERNEL_ARGS_FAST_PIXEL_FILTER \
		, __global void *samplerSharedDataBuff \
		, __global void *samplesBuff \
		, __global float *samplesDataBuff \
		, __global SampleResult *sampleResultsBuff \
		, __global EyePathInfo *eyePathInfos \
		KERNEL_ARGS_VOLUMES \
		, __global Ray *rays \
		, __global RayHit *rayHits \
		/* Film parameters */ \
		KERNEL_ARGS_FILM \
		/* Scene parameters */ \
		KERNEL_ARGS_INFINITELIGHTS \
		, __global const Material* restrict mats \
		, __global const MaterialEvalOp* restrict matEvalOps \
		, __global float *matEvalStacks \
		, const uint maxMaterialEvalStackSize \
		, __global const Texture* restrict texs \
		, __global const TextureEvalOp* restrict texEvalOps \
		, __global float *texEvalStacks \
		, const uint maxTextureEvalStackSize \
		, __global const SceneObject* restrict sceneObjs \
		, __global const ExtMesh* restrict meshDescs \
		, __global const Point* restrict vertices \
		KERNEL_ARGS_NORMALS_BUFFER \
		KERNEL_ARGS_TRINORMALS_BUFFER \
		KERNEL_ARGS_UVS_BUFFER \
		KERNEL_ARGS_COLS_BUFFER \
		KERNEL_ARGS_ALPHAS_BUFFER \
		KERNEL_ARGS_VERTEXAOVS_BUFFER \
		KERNEL_ARGS_TRIAOVS_BUFFER \
		, __global const Triangle* restrict triangles \
		, __global const InterpolatedTransform* restrict interpolatedTransforms \
		, __global const Camera* restrict camera \
		, __global const float* restrict cameraBokehDistribution \
		/* Lights */ \
		, __global const LightSource* restrict lights \
		KERNEL_ARGS_ENVLIGHTS \
		, __global const uint* restrict lightIndexOffsetByMeshIndex \
		, __global const uint* restrict lightIndexByTriIndex \
		KERNEL_ARGS_INFINITELIGHT \
		, __global const float* restrict lightsDistribution \
		, __global const float* restrict infiniteLightSourcesDistribution \
		, __global const DLSCacheEntry* restrict dlscAllEntries \
		, __global const float* restrict dlscDistributions \
		, __global const IndexBVHArrayNode* restrict dlscBVHNodes \
		, const float dlscRadius2 \
		, const float dlscNormalCosAngle \
		, __global const ELVCacheEntry* restrict elvcAllEntries \
		, __global const float* restrict elvcDistributions \
		, __global const uint* restrict elvcTileDistributionOffsets \
		, __global const IndexBVHArrayNode* restrict elvcBVHNodes \
		, const float elvcRadius2 \
		, const float elvcNormalCosAngle \
		, const uint elvcTilesXCount \
		, const uint elvcTilesYCount \
		/* Images */ \
		KERNEL_ARGS_IMAGEMAPS_PAGES \
		KERNEL_ARGS_PHOTONGI