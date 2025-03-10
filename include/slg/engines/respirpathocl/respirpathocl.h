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

#ifndef _SLG_RESPIRPATHOCL_H
#define	_SLG_RESPIRPATHOCL_H

#if !defined(LUXRAYS_DISABLE_OPENCL)

#include "slg/engines/pathocl/pathocl.h"
#include "slg/engines/pathoclbase/pathoclbase.h"

namespace slg {

class RespirPathOCLRenderEngine;

class RespirPathOCLRenderThread : public PathOCLOpenCLRenderThread {
public:
	RespirPathOCLRenderThread(const u_int index, luxrays::HardwareIntersectionDevice *device,
			RespirPathOCLRenderEngine *re);
	~RespirPathOCLRenderThread();

	virtual void StartRenderThread() override;
	virtual void Stop() override;

	friend class RespirPathOCLRenderEngine;
protected:
	void GetKernelParameters(std::vector<std::string> &params,
			luxrays::HardwareIntersectionDevice *intersectionDevice,
			const std::string renderEngineType,
			const float epsilonMin, const float epsilonMax) override;
	void InitRender() override;
	void InitGPUTaskBuffer(const u_int taskCount) override;
	void InitGPUTaskStateBuffer(const u_int taskCount) override;
	void InitGPUTaskBuffer() override;
	void GetThreadFilmSize(u_int *filmWidth, u_int *filmHeight, u_int *filmSubRegion) override;
	void RenderThreadImpl() override;
    void InitKernels() override;
	void InitPixelIndexMapBuffer(const u_int filmWidth, const u_int filmHeight);
	void InitRespirBuffers();
    void SetInitKernelArgs(const u_int filmIndex) override;
    void SetAdvancePathsKernelArgs(luxrays::HardwareDeviceKernel *advancePathsKernel, const u_int filmIndex) override;
    void SetAllAdvancePathsKernelArgs(const u_int filmIndex) override;
	void SetSpatialReuseKernelArgs(luxrays::HardwareDeviceKernel *spatialReuseKernel, const u_int filmIndex);
	void SetAllSpatialReuseKernelArgs(const u_int filmIndex);
	void SetKernelArgs() override;

    void EnqueueAdvancePathsKernel() override;
	bool CheckSyncedPathStates(ocl::pathoclbase::RespirGPUTaskState* tasksStateReadBuffer, const u_int taskCount, ocl::pathoclbase::PathState targetState);


	luxrays::HardwareDeviceBuffer* pixelIndexMapBuff;

	luxrays::HardwareDeviceKernel* spatialReuseKernel_MK_INIT;
	luxrays::HardwareDeviceKernel* spatialReuseKernel_MK_RESAMPLE_NEIGHBOR;
	luxrays::HardwareDeviceKernel* spatialReuseKernel_MK_CHECK_VISIBILITY;
	luxrays::HardwareDeviceKernel* spatialReuseKernel_MK_FINISH_ITERATION;
	luxrays::HardwareDeviceKernel* spatialReuseKernel_MK_FINISH_REUSE;
	luxrays::HardwareDeviceKernel* spatialReuseKernel_MK_SET_SPLAT;

	size_t spatialReuseResamplingVisibilityWorkGroupSize;
	size_t spatialReuseWorkGroupSize;
	const u_int spatialRadius;
};

//------------------------------------------------------------------------------
// Respir path tracing 100% OpenCL render engine
//------------------------------------------------------------------------------

class RespirPathOCLRenderEngine : public PathOCLRenderEngine {
public:
	RespirPathOCLRenderEngine(const RenderConfig *cfg);
	~RespirPathOCLRenderEngine();

	virtual RenderEngineType GetType() const { return GetObjectType(); }
	virtual std::string GetTag() const { return GetObjectTag(); }

	//--------------------------------------------------------------------------
	// Static methods used by RenderEngineRegistry
	//--------------------------------------------------------------------------

	static RenderEngineType GetObjectType() { return RESPIRPATHOCL; }
	static std::string GetObjectTag() { return "RESPIRPATHOCL"; }
	static luxrays::Properties ToProperties(const luxrays::Properties &cfg);
	static RenderEngine *FromProperties(const RenderConfig *rcfg);

	friend class RespirPathOCLRenderThread;
    friend class PathOCLNativeRenderThread;
    // TODO: replace above line with below if additional native thread functionality required
	// friend class RespirPathNativeRenderThread; 

	// number of times to iterate spatial reuse pass per frame
	u_int numSpatialReuseIterations;
	u_int spatialRadius;

protected:
	static const luxrays::Properties &GetDefaultProps();

	virtual PathOCLBaseOCLRenderThread *CreateOCLThread(const u_int index,
		luxrays::HardwareIntersectionDevice *device);

	void UpdateTaskCount() override;
};

}

#endif

#endif	/* _SLG_RESPIRPATHOCL_H */