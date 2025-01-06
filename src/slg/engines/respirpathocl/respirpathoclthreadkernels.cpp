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

void RespirPathOCLRenderThread::InitGPUTaskBuffer(const u_int taskCount) {
	intersectionDevice->AllocBufferRW(&tasksBuff, nullptr, sizeof(slg::ocl::pathoclbase::RespirGPUTask) * taskCount, "ReSPIR GPUTask");
}

void RespirPathOCLRenderThread::InitGPUTaskStateBuffer(const u_int taskCount) {
	intersectionDevice->AllocBufferRW(&tasksStateBuff, nullptr, sizeof(slg::ocl::pathoclbase::RespirGPUTaskState) * taskCount, "ReSPIR GPUTaskState");
}

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
	CompileKernel(intersectionDevice, program, &spatialReuseInitKernel, &workGroupSize,
			"spatialReuse_Init");
	CompileKernel(intersectionDevice, program, &spatialReuseIterateKernel, &workGroupSize,
			"spatialReuse_Iterate");
	CompileKernel(intersectionDevice, program, &spatialReuseDoneKernel, &workGroupSize,
			"spatialReuse_Done");
	CompileKernel(intersectionDevice, program, &spatialReuseSetSplatKernel, &workGroupSize,
			"spatialReuse_SetSplat");
	const double tSpatialReuseEnd = WallClockTime();
	SLG_LOG("[RespirPathOCLThread::" << threadIndex << "] Spatial reuse kernels compilation time: " << int((tSpatialReuseEnd - tSpatialReuseStart) * 1000.0) << "ms");

	delete program;

}

void RespirPathOCLRenderThread::SetInitKernelArgs(const u_int filmIndex) {
    PathOCLOpenCLRenderThread::SetInitKernelArgs(filmIndex);
}

void RespirPathOCLRenderThread::SetAdvancePathsKernelArgs(luxrays::HardwareDeviceKernel *advancePathsKernel, const u_int filmIndex) {
    PathOCLOpenCLRenderThread::SetAdvancePathsKernelArgs(advancePathsKernel, filmIndex);
}

void RespirPathOCLRenderThread::SetAllAdvancePathsKernelArgs(const u_int filmIndex) {
    PathOCLOpenCLRenderThread::SetAllAdvancePathsKernelArgs(filmIndex);
	if (spatialReuseInitKernel)
		SetAdvancePathsKernelArgs(spatialReuseInitKernel, filmIndex);
	if (spatialReuseIterateKernel)
		SetAdvancePathsKernelArgs(spatialReuseIterateKernel, filmIndex);
	if (spatialReuseDoneKernel)
		SetAdvancePathsKernelArgs(spatialReuseDoneKernel, filmIndex);
	if (spatialReuseSetSplatKernel)
		SetAdvancePathsKernelArgs(spatialReuseSetSplatKernel, filmIndex);
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
