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

#include "luxrays/core/geometry/transform.h"
#include "luxrays/core/randomgen.h"
#include "luxrays/utils/ocl.h"
#include "luxrays/devices/ocldevice.h"
#include "luxrays/kernels/kernels.h"

#include "slg/slg.h"
#include "slg/kernels/kernels.h"
#include "slg/renderconfig.h"
#include "slg/engines/pathocl/pathocl.h"
#include "slg/engines/respirpathocl/respirpathocl.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// RespirPathOCLRenderThread
//------------------------------------------------------------------------------

RespirPathOCLRenderThread::RespirPathOCLRenderThread(const u_int index, luxrays::HardwareIntersectionDevice *device,
        PathOCLRenderEngine *re)
    : PathOCLOpenCLRenderThread(index, device, re) {

}

RespirPathOCLRenderThread::~RespirPathOCLRenderThread() {

}

void RespirPathOCLRenderThread::StartRenderThread() {
    PathOCLOpenCLRenderThread::StartRenderThread();
}

void RespirPathOCLRenderThread::GetThreadFilmSize(u_int *filmWidth, u_int *filmHeight, u_int *filmSubRegion) {
    PathOCLOpenCLRenderThread::GetThreadFilmSize(filmWidth, filmHeight, filmSubRegion);
}

void RespirPathOCLRenderThread::RenderThreadImpl() {
    SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Rendering thread started");

	PathOCLRenderEngine *engine = (PathOCLRenderEngine *)renderEngine;
	const u_int taskCount = engine->taskCount;

	intersectionDevice->PushThreadCurrentDevice();

    // Keep track of rendered samples per pixel for this thread.
    // TODO: remove, since this should already be tracked in the film class I think
    u_int spp = 0;

	try {
		//----------------------------------------------------------------------
		// Execute initialization kernels
		//----------------------------------------------------------------------

		// Clear the frame buffer
		const u_int filmPixelCount = threadFilms[0]->film->GetWidth() * threadFilms[0]->film->GetHeight();
		intersectionDevice->EnqueueKernel(filmClearKernel,
			HardwareDeviceRange(RoundUp<u_int>(filmPixelCount, filmClearWorkGroupSize)),
			HardwareDeviceRange(filmClearWorkGroupSize));

		// Initialize random number generator seeds
		intersectionDevice->EnqueueKernel(initSeedKernel,
				HardwareDeviceRange(engine->taskCount), HardwareDeviceRange(initWorkGroupSize));

		// Initialize the tasks buffer
		intersectionDevice->EnqueueKernel(initKernel,
				HardwareDeviceRange(engine->taskCount), HardwareDeviceRange(initWorkGroupSize));

		// Check if I have to load the start film
		if (engine->hasStartFilm && (threadIndex == 0))
			threadFilms[0]->SendFilm(intersectionDevice);

        slg::ocl::pathoclbase::RespirGPUTaskState* tasksState = (slg::ocl::pathoclbase::RespirGPUTaskState*)malloc(sizeof(*tasksState) * taskCount);

		//----------------------------------------------------------------------
		// Rendering loop
		//----------------------------------------------------------------------

		// The film refresh time target
		const double targetTime = 0.2; // 200ms

        // TODO: Let this be configurable
        const u_int numSpatialReuseIterations = 1;

		u_int iterations = 4;
		u_int totalIterations = 0;

		double totalTransferTime = 0.0;
		double totalKernelTime = 0.0;

		while (!boost::this_thread::interruption_requested()) {
			//if (threadIndex == 0)
			//	SLG_LOG("[DEBUG] =================================");

			// Check if we are in pause mode
			if (engine->pauseMode) {
				// Check every 100ms if I have to continue the rendering
				while (!boost::this_thread::interruption_requested() && engine->pauseMode)
					boost::this_thread::sleep(boost::posix_time::millisec(100));

				if (boost::this_thread::interruption_requested())
					break;
			}

			//------------------------------------------------------------------

			const double timeTransferStart = WallClockTime();

			// Transfer the film only if I have already spent enough time running
			// rendering kernels. This is very important when rendering very high
			// resolution images (for instance at 4961x3508)

			if (totalTransferTime < totalKernelTime * (1.0 / 100.0)) {
				// Async. transfer of the Film buffers
				threadFilms[0]->RecvFilm(intersectionDevice);

				// Async. transfer of GPU task statistics
				intersectionDevice->EnqueueReadBuffer(
					taskStatsBuff,
					CL_FALSE,
					sizeof(slg::ocl::pathoclbase::GPUTaskStats) * taskCount,
					gpuTaskStats);

				intersectionDevice->FinishQueue();
				
				// I need to update the film samples count
				
				double totalCount = 0.0;
				for (size_t i = 0; i < taskCount; ++i)
					totalCount += gpuTaskStats[i].sampleCount;
				threadFilms[0]->film->SetSampleCount(totalCount, totalCount, 0.0);

				//SLG_LOG("[DEBUG] film transferred");
			}
			const double timeTransferEnd = WallClockTime();
			totalTransferTime += timeTransferEnd - timeTransferStart;

			//------------------------------------------------------------------
			
			const double timeKernelStart = WallClockTime();

			// This is required for updating film denoiser parameter
			if (threadFilms[0]->film->GetDenoiser().IsEnabled()) {
				boost::unique_lock<boost::mutex> lock(engine->setKernelArgsMutex);
				SetAllAdvancePathsKernelArgs(0);
			}

            SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Beginning rendering for frame " << spp << ".");

            // Generate camera rays for each pixel in this frame.
            intersectionDevice->EnqueueKernel(advancePathsKernel_MK_GENERATE_CAMERA_RAY,
			    HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));

			// Get next sample if this is not the first iteration of this loop.
	        intersectionDevice->EnqueueKernel(advancePathsKernel_MK_NEXT_SAMPLE,
			    HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));

            // Perform initial path resampling to get canonical samples for each pixel this frame.
            bool isInitialPathResamplingDone = false;
            u_int totalIterationsThisFrame = 0;
			u_int iterations = 4;
            while (!isInitialPathResamplingDone) {
				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Queuing advance paths kernels");

                // Trace until all paths are completed for this frame.
                for (u_int i = 0; i < iterations; ++i) {
                    // Trace rays
                    intersectionDevice->EnqueueTraceRayBuffer(raysBuff, hitsBuff, taskCount);

                    // Advance to next path state
                    EnqueueAdvancePathsKernel();
                }

				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Finished queuing advance paths kernels");

                // Wait for all kernels to finish running.
			    intersectionDevice->FinishQueue();

				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] All advance paths kernels finished execution");

                // Check if initial path resampling for all pixels is complete
                // TODO: move pathState to a separate buffer so minimal amount of memory needs to be read here
                intersectionDevice->EnqueueReadBuffer(tasksStateBuff, true,
                    sizeof(slg::ocl::pathoclbase::RespirGPUTaskState) * taskCount,
                    tasksState);

				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Checking whether initial path resampling is complete");

				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Task count is " << taskCount);
                isInitialPathResamplingDone = true;
                for (u_int i = 0; i < taskCount; i++) {
					//SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] TaskState(" << i << ") PathState=" << tasksState[i].state);
                    if (tasksState[i].state != slg::ocl::pathoclbase::PathState::SYNC) {
                        isInitialPathResamplingDone = false;
                        break;
                    }
                }

				if (isInitialPathResamplingDone) {
					break;
				}

                totalIterations += iterations;
                totalIterationsThisFrame += iterations;
                // TODO: redo this logic outside of the loop later
                //      can compare the number of iterations it takes for each frame
                const double timeKernelEnd = WallClockTime();
                totalKernelTime += timeKernelEnd - timeKernelStart;

                // Check if I have to adjust the number of kernel enqueued
                if (timeKernelEnd - timeKernelStart > targetTime)
                    iterations = Max<u_int>(iterations - 1, 1);
                else
                    iterations = Min<u_int>(iterations + 1, 128);
            }

			SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Initial path resampling is complete, performing spatial reuse");
            
            // Perform spatial reuse.
			if (numSpatialReuseIterations > 0) {
				// Initialize spatial reuse iterations
				intersectionDevice->EnqueueKernel(spatialReuseInitKernel,
						HardwareDeviceRange(engine->taskCount), HardwareDeviceRange(initWorkGroupSize));
				// Ensure all paths are synced before continuing
				intersectionDevice->FinishQueue();

				// Iterate x times
				for (u_int i = 0; i < numSpatialReuseIterations; i++) {
					// Select neighboring pixels to resample from.
					intersectionDevice->EnqueueKernel(spatialReuseIterateKernel,
						HardwareDeviceRange(engine->taskCount), HardwareDeviceRange(initWorkGroupSize));
					
					// TODO: When in shift mapping mode, do not need to perform MK_ILLUMINATE calculations.
					// Only need to advance the random number by the amount it would use.
					
					// Since spatial reuse (shift mapping step) needs to retrace only paths already traced in this frame, 
					// the max number of iterations possible is the amount required for 
					// the path that took the most number of iterations in this frame.
					for (u_int i = 0; i < totalIterationsThisFrame; ++i) {
						// Trace rays
						intersectionDevice->EnqueueTraceRayBuffer(raysBuff, hitsBuff, taskCount);

						// Advance to next path state
						EnqueueAdvancePathsKernel();
					}

					// Ensure all paths are synced before continuing
					intersectionDevice->FinishQueue();
				}
				// Set up for splatting
				intersectionDevice->EnqueueKernel(spatialReuseDoneKernel,
						HardwareDeviceRange(engine->taskCount), HardwareDeviceRange(initWorkGroupSize));
			}

            // Splat pixels.
			intersectionDevice->EnqueueKernel(spatialReuseSetSplatKernel,
                    HardwareDeviceRange(engine->taskCount), HardwareDeviceRange(initWorkGroupSize));
            intersectionDevice->EnqueueKernel(advancePathsKernel_MK_SPLAT_SAMPLE,
			    HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));

            spp++;

			/*if (threadIndex == 0)
				SLG_LOG("[DEBUG] transfer time: " << (timeTransferEnd - timeTransferStart) * 1000.0 << "ms "
						"kernel time: " << (timeKernelEnd - timeKernelStart) * 1000.0 << "ms "
						"iterations: " << iterations << " #"<< taskCount << ")");*/

			// Check halt conditions
			if (engine->film->GetConvergence() == 1.f)
				break;
		}
        free(tasksState);

	} catch (boost::thread_interrupted) {
		SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] Rendering thread halted");
	}
    SLG_LOG("[PathOCLRespirRenderThread::" << threadIndex << "]: Rendered " << spp << "samples per pixel (spp) in total.");

	threadFilms[0]->RecvFilm(intersectionDevice);
	intersectionDevice->FinishQueue();
	
	threadDone = true;
	
	intersectionDevice->PopThreadCurrentDevice();
}

#endif
