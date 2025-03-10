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

#if !defined(LUXRAYS_DISABLE_OPENCL)

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "luxrays/core/geometry/transform.h"
#include "luxrays/core/randomgen.h"
#include "luxrays/utils/ocl.h"
#include "luxrays/devices/ocldevice.h"
#include "luxrays/kernels/kernels.h"

#include "slg/slg.h"
#include "slg/kernels/kernels.h"
#include "slg/renderconfig.h"
#include "slg/engines/respirpathocl/respirpathocl.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// RespirPathOCLRenderThread kernel functions
//------------------------------------------------------------------------------

void RespirPathOCLRenderThread::GetKernelParameters(
		vector<string> &params,
		HardwareIntersectionDevice *intersectionDevice,
		const string renderEngineType,
		const float epsilonMin, const float epsilonMax) {
	params.push_back("-D LUXRAYS_OPENCL_KERNEL");
	params.push_back("-D SLG_OPENCL_KERNEL");
	params.push_back("-D RENDER_ENGINE_" + renderEngineType);
	params.push_back("-D PARAM_RAY_EPSILON_MIN=" + ToString(epsilonMin) + "f");
	params.push_back("-D PARAM_RAY_EPSILON_MAX=" + ToString(epsilonMax) + "f");

	const OpenCLDeviceDescription *oclDeviceDesc = dynamic_cast<const OpenCLDeviceDescription *>(intersectionDevice->GetDeviceDesc());
	if (oclDeviceDesc) {
		if (oclDeviceDesc->IsAMDPlatform())
			params.push_back("-D LUXCORE_AMD_OPENCL");
		else if (oclDeviceDesc->IsNVIDIAPlatform())
			params.push_back("-D LUXCORE_NVIDIA_OPENCL");
		else
			params.push_back("-D LUXCORE_GENERIC_OPENCL");
	}}

void RespirPathOCLRenderThread::InitKernels() {
	//--------------------------------------------------------------------------
	// Compile kernels
	//--------------------------------------------------------------------------

	const double tStart = WallClockTime();

	// A safety check
	switch (intersectionDevice->GetAccelerator()->GetType()) {
		case ACCEL_BVH:
			break;
		case ACCEL_MBVH:
			break;
		case ACCEL_EMBREE:
		throw runtime_error("EMBREE accelerator is not supported in PathOCLBaseRenderThread::InitKernels()");
		case ACCEL_OPTIX:
			break;
		default:
			throw runtime_error("Unknown accelerator in PathOCLBaseRenderThread::InitKernels()");
	}

	vector<string> kernelsParameters;
	GetKernelParameters(kernelsParameters, intersectionDevice,
			RenderEngine::RenderEngineType2String(renderEngine->GetType()),
			MachineEpsilon::GetMin(), MachineEpsilon::GetMax());

	const string kernelSource = GetKernelSources();

	if (renderEngine->writeKernelsToFile) {
		// Some debug code to write the OpenCL kernel source to a file
		const string kernelFileName = "kernel_source_device_" + ToString(threadIndex) + ".cl";
		ofstream kernelFile(kernelFileName.c_str());
		string kernelDefs = oclKernelPersistentCache::ToOptsString(kernelsParameters);
		boost::replace_all(kernelDefs, "-D", "\n#define");
		boost::replace_all(kernelDefs, "=", " ");
		kernelFile << kernelDefs << endl << endl << kernelSource << endl;
		kernelFile.close();
	}

	if ((renderEngine->additionalOpenCLKernelOptions.size() > 0) &&
			(intersectionDevice->GetDeviceDesc()->GetType() & DEVICE_TYPE_OPENCL_ALL))
		kernelsParameters.insert(kernelsParameters.end(), renderEngine->additionalOpenCLKernelOptions.begin(), renderEngine->additionalOpenCLKernelOptions.end());
	if ((renderEngine->additionalCUDAKernelOptions.size() > 0) &&
			(intersectionDevice->GetDeviceDesc()->GetType() & DEVICE_TYPE_CUDA_ALL))
		kernelsParameters.insert(kernelsParameters.end(), renderEngine->additionalCUDAKernelOptions.begin(), renderEngine->additionalCUDAKernelOptions.end());

	// Build the kernel source/parameters hash
	const string newKernelSrcHash = oclKernelPersistentCache::HashString(oclKernelPersistentCache::ToOptsString(kernelsParameters))
			+ "-" +
			oclKernelPersistentCache::HashString(kernelSource);
	if (newKernelSrcHash == kernelSrcHash) {
		// There is no need to re-compile the kernel
		return;
	} else
		kernelSrcHash = newKernelSrcHash;

	SLG_LOG("[PathOCLBaseRenderThread::" << threadIndex << "] Compiling kernels ");

	HardwareDeviceProgram *program = nullptr;
	intersectionDevice->CompileProgram(&program, kernelsParameters, kernelSource, "PathOCL kernel");

	// Film clear kernel
	CompileKernel(intersectionDevice, program, &filmClearKernel, &filmClearWorkGroupSize, "Film_Clear");

	// Init kernel

	CompileKernel(intersectionDevice, program, &initSeedKernel, &initWorkGroupSize, "InitSeed");
	CompileKernel(intersectionDevice, program, &initKernel, &initWorkGroupSize, "Init");

	// AdvancePaths kernel (Micro-Kernels)

	size_t workGroupSize;
	CompileKernel(intersectionDevice, program, &advancePathsKernel_MK_RT_NEXT_VERTEX, &advancePathsWorkGroupSize,
			"AdvancePaths_MK_RT_NEXT_VERTEX");
	CompileKernel(intersectionDevice, program, &advancePathsKernel_MK_HIT_NOTHING, &workGroupSize,
			"AdvancePaths_MK_HIT_NOTHING");
	advancePathsWorkGroupSize = Min(advancePathsWorkGroupSize, workGroupSize);
	CompileKernel(intersectionDevice, program, &advancePathsKernel_MK_HIT_OBJECT, &workGroupSize,
			"AdvancePaths_MK_HIT_OBJECT");
	advancePathsWorkGroupSize = Min(advancePathsWorkGroupSize, workGroupSize);
	CompileKernel(intersectionDevice, program, &advancePathsKernel_MK_RT_DL, &workGroupSize,
			"AdvancePaths_MK_RT_DL");
	advancePathsWorkGroupSize = Min(advancePathsWorkGroupSize, workGroupSize);
	CompileKernel(intersectionDevice, program, &advancePathsKernel_MK_DL_ILLUMINATE, &workGroupSize,
			"AdvancePaths_MK_DL_ILLUMINATE");
	advancePathsWorkGroupSize = Min(advancePathsWorkGroupSize, workGroupSize);
	CompileKernel(intersectionDevice, program, &advancePathsKernel_MK_DL_SAMPLE_BSDF, &workGroupSize,
			"AdvancePaths_MK_DL_SAMPLE_BSDF");
	advancePathsWorkGroupSize = Min(advancePathsWorkGroupSize, workGroupSize);
	CompileKernel(intersectionDevice, program, &advancePathsKernel_MK_GENERATE_NEXT_VERTEX_RAY, &workGroupSize,
			"AdvancePaths_MK_GENERATE_NEXT_VERTEX_RAY");
	advancePathsWorkGroupSize = Min(advancePathsWorkGroupSize, workGroupSize);
	CompileKernel(intersectionDevice, program, &advancePathsKernel_MK_SPLAT_SAMPLE, &workGroupSize,
			"AdvancePaths_MK_SPLAT_SAMPLE");
	advancePathsWorkGroupSize = Min(advancePathsWorkGroupSize, workGroupSize);
	CompileKernel(intersectionDevice, program, &advancePathsKernel_MK_NEXT_SAMPLE, &workGroupSize,
			"AdvancePaths_MK_NEXT_SAMPLE");
	advancePathsWorkGroupSize = Min(advancePathsWorkGroupSize, workGroupSize);
	CompileKernel(intersectionDevice, program, &advancePathsKernel_MK_GENERATE_CAMERA_RAY, &workGroupSize,
			"AdvancePaths_MK_GENERATE_CAMERA_RAY");
	advancePathsWorkGroupSize = Min(advancePathsWorkGroupSize, workGroupSize);
	SLG_LOG("[PathOCLBaseRenderThread::" << threadIndex << "] AdvancePaths_MK_* workgroup size: " << advancePathsWorkGroupSize);
	const double tEnd = WallClockTime();
	SLG_LOG("[PathOCLBaseRenderThread::" << threadIndex << "] Kernels compilation time: " << int((tEnd - tStart) * 1000.0) << "ms");

	const double tSpatialReuseStart = WallClockTime();
	SLG_LOG("[RespirPathOCLThread::" << threadIndex << "] Compiling kernels ");

	CompileKernel(intersectionDevice, program, &spatialReuseKernel_MK_RESAMPLE_NEIGHBOR, &spatialReuseResamplingVisibilityWorkGroupSize,
			"SpatialReuse_ResampleNeighbor");
	CompileKernel(intersectionDevice, program, &spatialReuseKernel_MK_CHECK_VISIBILITY, &workGroupSize,
		"SpatialReuse_CheckVisibility");
	spatialReuseResamplingVisibilityWorkGroupSize = Min(spatialReuseResamplingVisibilityWorkGroupSize, workGroupSize);
	SLG_LOG("[PathOCLBaseRenderThread::" << threadIndex << "] Spatial Reuse resampling and visibility kernels workgroup size: " << spatialReuseResamplingVisibilityWorkGroupSize);

	CompileKernel(intersectionDevice, program, &spatialReuseKernel_MK_INIT, &spatialReuseWorkGroupSize,
		"SpatialReuse_Init");
	CompileKernel(intersectionDevice, program, &spatialReuseKernel_MK_FINISH_ITERATION, &workGroupSize,
			"SpatialReuse_FinishIteration");
	spatialReuseWorkGroupSize = Min(spatialReuseWorkGroupSize, workGroupSize);
	CompileKernel(intersectionDevice, program, &spatialReuseKernel_MK_FINISH_REUSE, &workGroupSize,
			"SpatialReuse_FinishReuse");
	spatialReuseWorkGroupSize = Min(spatialReuseWorkGroupSize, workGroupSize);
	CompileKernel(intersectionDevice, program, &spatialReuseKernel_MK_SET_SPLAT, &workGroupSize,
			"SpatialReuse_SetSplat");
	spatialReuseWorkGroupSize = Min(spatialReuseWorkGroupSize, workGroupSize);
	SLG_LOG("[PathOCLBaseRenderThread::" << threadIndex << "] Spatial Reuse general kernels workgroup size: " << spatialReuseWorkGroupSize);
	const double tSpatialReuseEnd = WallClockTime();
	SLG_LOG("[RespirPathOCLThread::" << threadIndex << "] Spatial reuse kernels compilation time: " << int((tSpatialReuseEnd - tSpatialReuseStart) * 1000.0) << "ms");

	delete program;
}

void RespirPathOCLRenderThread::InitGPUTaskBuffer(const u_int taskCount) {
	intersectionDevice->AllocBufferRW(&tasksBuff, nullptr, sizeof(slg::ocl::pathoclbase::RespirGPUTask) * taskCount, "ReSPIR GPUTask");
}

void RespirPathOCLRenderThread::InitGPUTaskStateBuffer(const u_int taskCount) {
	intersectionDevice->AllocBufferRW(&tasksStateBuff, nullptr, sizeof(slg::ocl::pathoclbase::RespirGPUTaskState) * taskCount, "ReSPIR GPUTaskState");
}

void RespirPathOCLRenderThread::InitGPUTaskBuffer() {
	PathOCLBaseOCLRenderThread::InitGPUTaskBuffer();
}

void RespirPathOCLRenderThread::InitPixelIndexMapBuffer(const u_int filmWidth, const u_int filmHeight) {
	size_t size = sizeof(int) * filmWidth * filmHeight;
	intersectionDevice->AllocBufferRW(&pixelIndexMapBuff, nullptr, size, "PixelIndexMap");
	SLG_LOG("Initial pixelindexmap dims " << filmHeight << " " << filmWidth);
	int initial[filmHeight][filmWidth]{};
	for (int i = 0; i < filmHeight; i++) {
		for (int j = 0; j < filmWidth; j++) {
			initial[i][j] = -1;
		}
	}
	intersectionDevice->EnqueueWriteBuffer(pixelIndexMapBuff, CL_TRUE, size, initial);
}

void RespirPathOCLRenderThread::InitRespirBuffers() {
	InitPixelIndexMapBuffer(threadFilms[0]->film->GetWidth(), threadFilms[0]->film->GetHeight());
}

void RespirPathOCLRenderThread::SetInitKernelArgs(const u_int filmIndex) {
    PathOCLOpenCLRenderThread::SetInitKernelArgs(filmIndex);
}

void RespirPathOCLRenderThread::SetAdvancePathsKernelArgs(luxrays::HardwareDeviceKernel *advancePathsKernel, const u_int filmIndex) {
    PathOCLOpenCLRenderThread::SetAdvancePathsKernelArgs(advancePathsKernel, filmIndex);
}

void RespirPathOCLRenderThread::SetAllAdvancePathsKernelArgs(const u_int filmIndex) {
    PathOCLOpenCLRenderThread::SetAllAdvancePathsKernelArgs(filmIndex);
}

void RespirPathOCLRenderThread::SetSpatialReuseKernelArgs(HardwareDeviceKernel *spatialReuseKernel, const u_int filmIndex) {
	CompiledScene *cscene = renderEngine->compiledScene;

	u_int argIndex = 0;
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, taskConfigBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, tasksBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, tasksDirectLightBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, tasksStateBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, taskStatsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, pixelFilterBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, samplerSharedDataBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, samplesBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, sampleDataBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, sampleResultsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, eyePathInfosBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, directLightVolInfosBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, raysBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, hitsBuff);

	// Film parameters
	argIndex = threadFilms[filmIndex]->SetFilmKernelArgs(intersectionDevice, spatialReuseKernel, argIndex);

	// Scene parameters
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->worldBSphere.center.x);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->worldBSphere.center.y);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->worldBSphere.center.z);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->worldBSphere.rad);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, materialsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, materialEvalOpsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, materialEvalStackBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->maxMaterialEvalStackSize);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, texturesBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, textureEvalOpsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, textureEvalStackBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->maxTextureEvalStackSize);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, scnObjsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, meshDescsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, vertsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, normalsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, triNormalsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, uvsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, colsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, alphasBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, vertexAOVBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, triAOVBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, trianglesBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, interpolatedTransformsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cameraBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cameraBokehDistributionBuff);
	// Lights
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, lightsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, envLightIndicesBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, (u_int)cscene->envLightIndices.size());
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, lightIndexOffsetByMeshIndexBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, lightIndexByTriIndexBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, envLightDistributionsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, lightsDistributionBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, infiniteLightSourcesDistributionBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, dlscAllEntriesBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, dlscDistributionsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, dlscBVHNodesBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->dlscRadius2);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->dlscNormalCosAngle);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, elvcAllEntriesBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, elvcDistributionsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, elvcTileDistributionOffsetsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, elvcBVHNodesBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->elvcRadius2);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->elvcNormalCosAngle);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->elvcTilesXCount);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->elvcTilesYCount);

	// Images
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, imageMapDescsBuff);
	for (u_int i = 0; i < 8; ++i) {
		if (i < imageMapsBuff.size())
			intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, imageMapsBuff[i]);
		else
			intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, nullptr);
	}

	// PhotonGI cache
	// TODO: remove this since this Respir implementation doesn't use photonGI caching
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, pgicRadiancePhotonsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, cscene->pgicLightGroupCounts);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, pgicRadiancePhotonsValuesBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, pgicRadiancePhotonsBVHNodesBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, pgicCausticPhotonsBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, pgicCausticPhotonsBVHNodesBuff);

	// Spatial reuse
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, pixelIndexMapBuff);
	intersectionDevice->SetKernelArg(spatialReuseKernel, argIndex++, spatialRadius);
}

void RespirPathOCLRenderThread::SetAllSpatialReuseKernelArgs(const u_int filmIndex) {
	// TODO: optimize performance by setting smaller kernel args for spatial reuse kernels
	if (spatialReuseKernel_MK_INIT)
		SetSpatialReuseKernelArgs(spatialReuseKernel_MK_INIT, filmIndex);
	if (spatialReuseKernel_MK_RESAMPLE_NEIGHBOR)
		SetSpatialReuseKernelArgs(spatialReuseKernel_MK_RESAMPLE_NEIGHBOR, filmIndex);
	if (spatialReuseKernel_MK_CHECK_VISIBILITY)
		SetSpatialReuseKernelArgs(spatialReuseKernel_MK_CHECK_VISIBILITY, filmIndex);
	if (spatialReuseKernel_MK_FINISH_ITERATION)
		SetSpatialReuseKernelArgs(spatialReuseKernel_MK_FINISH_ITERATION, filmIndex);
	if (spatialReuseKernel_MK_FINISH_REUSE)
		SetSpatialReuseKernelArgs(spatialReuseKernel_MK_FINISH_REUSE, filmIndex);
	if (spatialReuseKernel_MK_SET_SPLAT)
		SetSpatialReuseKernelArgs(spatialReuseKernel_MK_SET_SPLAT, filmIndex);
}

void RespirPathOCLRenderThread::SetKernelArgs() {
	// Set init kernel args and advance paths kernels args
	PathOCLBaseOCLRenderThread::SetKernelArgs();

	SLG_LOG("[RespirPathOCLBaseRenderThread::" << threadIndex << "] Setting kernel arguments");

	//--------------------------------------------------------------------------
	// Spatial reuse kernels
	//--------------------------------------------------------------------------

	SLG_LOG("[PathOCLBaseRenderThread::" << threadIndex << "] Setting spatial reuse kernel arguments");

	SetAllSpatialReuseKernelArgs(0);
}

void RespirPathOCLRenderThread::InitRender() {
	//--------------------------------------------------------------------------
	// Film definition
	//--------------------------------------------------------------------------

	InitFilm();

	//--------------------------------------------------------------------------
	// Camera definition
	//--------------------------------------------------------------------------

	InitCamera();

	//--------------------------------------------------------------------------
	// Scene geometry
	//--------------------------------------------------------------------------

	InitGeometry();

	//--------------------------------------------------------------------------
	// Image maps
	//--------------------------------------------------------------------------

	InitImageMaps();

	//--------------------------------------------------------------------------
	// Texture definitions
	//--------------------------------------------------------------------------

	InitTextures();

	//--------------------------------------------------------------------------
	// Material definitions
	//--------------------------------------------------------------------------

	InitMaterials();

	//--------------------------------------------------------------------------
	// Mesh <=> Material links
	//--------------------------------------------------------------------------

	InitSceneObjects();

	//--------------------------------------------------------------------------
	// Light definitions
	//--------------------------------------------------------------------------

	InitLights();

	//--------------------------------------------------------------------------
	// Light definitions
	//--------------------------------------------------------------------------

	InitPhotonGI();

	//--------------------------------------------------------------------------
	// GPUTaskStats
	//--------------------------------------------------------------------------

	const u_int taskCount = renderEngine->taskCount;

	// In case renderEngine->taskCount has changed
	delete[] gpuTaskStats;
	gpuTaskStats = new slg::ocl::pathoclbase::GPUTaskStats[taskCount];
	for (u_int i = 0; i < taskCount; ++i)
		gpuTaskStats[i].sampleCount = 0;

	//--------------------------------------------------------------------------
	// Allocate Ray/RayHit buffers
	//--------------------------------------------------------------------------

	intersectionDevice->AllocBufferRW(&raysBuff, nullptr, sizeof(Ray) * taskCount, "Ray");
	intersectionDevice->AllocBufferRW(&hitsBuff, nullptr, sizeof(RayHit) * taskCount, "RayHit");

	//--------------------------------------------------------------------------
	// Allocate GPU task buffers
	//--------------------------------------------------------------------------

	InitGPUTaskBuffer();

	//--------------------------------------------------------------------------
	// Allocate GPU task statistic buffers
	//--------------------------------------------------------------------------

	intersectionDevice->AllocBufferRW(&taskStatsBuff, nullptr, sizeof(slg::ocl::pathoclbase::GPUTaskStats) * taskCount, "GPUTask Stats");

	//--------------------------------------------------------------------------
	// Allocate sampler shared data buffer
	//--------------------------------------------------------------------------

	InitSamplerSharedDataBuffer();

	//--------------------------------------------------------------------------
	// Allocate sample buffers
	//--------------------------------------------------------------------------

	InitSamplesBuffer();

	//--------------------------------------------------------------------------
	// Allocate sample data buffers
	//--------------------------------------------------------------------------

	InitSampleDataBuffer();

	//--------------------------------------------------------------------------
	// Allocate sample result buffers
	//--------------------------------------------------------------------------

	InitSampleResultsBuffer();

	//--------------------------------------------------------------------------
	// Allocate volume info buffers if required
	//--------------------------------------------------------------------------

	intersectionDevice->AllocBufferRW(&eyePathInfosBuff, nullptr, sizeof(slg::ocl::EyePathInfo) * taskCount, "PathInfo");

	//--------------------------------------------------------------------------
	// Allocate volume info buffers if required
	//--------------------------------------------------------------------------

	intersectionDevice->AllocBufferRW(&directLightVolInfosBuff, nullptr, sizeof(slg::ocl::PathVolumeInfo) * taskCount, "DirectLightVolumeInfo");

	//--------------------------------------------------------------------------
	// Allocate GPU pixel filter distribution
	//--------------------------------------------------------------------------

	intersectionDevice->AllocBufferRO(&pixelFilterBuff, renderEngine->pixelFilterDistribution,
			renderEngine->pixelFilterDistributionSize, "Pixel Filter Distribution");
	
	//--------------------------------------------------------------------------
	// Allocate Respir related buffers
	//--------------------------------------------------------------------------

	InitRespirBuffers();

	//--------------------------------------------------------------------------
	// Compile kernels
	//--------------------------------------------------------------------------

	InitKernels();

	//--------------------------------------------------------------------------
	// Initialize
	//--------------------------------------------------------------------------

	// Set kernel arguments
	SetKernelArgs();

	// Clear all thread films
	BOOST_FOREACH(ThreadFilm *threadFilm, threadFilms) {
		intersectionDevice->PushThreadCurrentDevice();
		threadFilm->ClearFilm(intersectionDevice, filmClearKernel, filmClearWorkGroupSize);
		intersectionDevice->PopThreadCurrentDevice();
	}

	intersectionDevice->FinishQueue();

	// Reset statistics in order to be more accurate
	intersectionDevice->ResetPerformaceStats();
}

void RespirPathOCLRenderThread::EnqueueAdvancePathsKernel() {
    const u_int taskCount = renderEngine->taskCount;

	// Micro kernels version
	intersectionDevice->EnqueueKernel(advancePathsKernel_MK_RT_NEXT_VERTEX,
			HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
	intersectionDevice->EnqueueKernel(advancePathsKernel_MK_HIT_NOTHING,
			HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
	intersectionDevice->EnqueueKernel(advancePathsKernel_MK_HIT_OBJECT,
			HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
	intersectionDevice->EnqueueKernel(advancePathsKernel_MK_RT_DL,
			HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
	intersectionDevice->EnqueueKernel(advancePathsKernel_MK_DL_ILLUMINATE,
			HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
	intersectionDevice->EnqueueKernel(advancePathsKernel_MK_DL_SAMPLE_BSDF,
			HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
	intersectionDevice->EnqueueKernel(advancePathsKernel_MK_GENERATE_NEXT_VERTEX_RAY,
			HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
}

#endif
