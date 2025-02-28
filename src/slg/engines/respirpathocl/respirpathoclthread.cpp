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
#include "slg/engines/respirpathocl/respirpathocl.h"

using namespace std;
using namespace luxrays;
using namespace slg;

//------------------------------------------------------------------------------
// RespirPathOCLRenderThread
//------------------------------------------------------------------------------

RespirPathOCLRenderThread::RespirPathOCLRenderThread(const u_int index, luxrays::HardwareIntersectionDevice *device,
        RespirPathOCLRenderEngine *re)
    : PathOCLOpenCLRenderThread(index, device, re) {
    spatialReuseInitKernel = nullptr;
	spatialReuseResampleNeighborKernel = nullptr;
	spatialReuseFinishIterationKernel = nullptr;
	spatialReuseSetSplatKernel = nullptr;
}

RespirPathOCLRenderThread::~RespirPathOCLRenderThread() {
    delete spatialReuseInitKernel;
	delete spatialReuseResampleNeighborKernel;
	delete spatialReuseFinishIterationKernel;
	delete spatialReuseSetSplatKernel;
}

void RespirPathOCLRenderThread::StartRenderThread() {
    PathOCLOpenCLRenderThread::StartRenderThread();
}

void RespirPathOCLRenderThread::GetThreadFilmSize(u_int *filmWidth, u_int *filmHeight, u_int *filmSubRegion) {
    PathOCLOpenCLRenderThread::GetThreadFilmSize(filmWidth, filmHeight, filmSubRegion);
}

// Check that all path states are equal to the target
bool RespirPathOCLRenderThread::CheckSyncedPathStates(slg::ocl::pathoclbase::RespirGPUTaskState* tasksStateReadBuffer, 
		const u_int taskCount, slg::ocl::pathoclbase::PathState targetState) {
	// TODO: move pathState to a separate buffer so minimal amount of memory needs to be read here
	intersectionDevice->EnqueueReadBuffer(tasksStateBuff, true,
		sizeof(slg::ocl::pathoclbase::RespirGPUTaskState) * taskCount,
		tasksStateReadBuffer);

	for (u_int i = 0; i < taskCount; i++) {
		//SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] TaskState(" << i << ") PathState=" << tasksState[i].state);
		if (tasksStateReadBuffer[i].state != targetState) {
			SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] TaskState(" << i << ") PathState=" << tasksStateReadBuffer[i].state << " Not synced.");
			return false;
		}
	}
	return true;
}

void RespirPathOCLRenderThread::RenderThreadImpl() {
    SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Rendering thread started");

	RespirPathOCLRenderEngine *engine = (RespirPathOCLRenderEngine *)renderEngine;
	const u_int taskCount = engine->taskCount;

	intersectionDevice->PushThreadCurrentDevice();

    // Keep track of rendered samples per pixel for this thread.
    // TODO: remove, since this should already be tracked in the film class I think
    u_int numFrames = 0;

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

        slg::ocl::pathoclbase::RespirGPUTaskState* tasksStateReadBuffer = (slg::ocl::pathoclbase::RespirGPUTaskState*)malloc(sizeof(*tasksStateReadBuffer) * taskCount);

		//----------------------------------------------------------------------
		// Rendering loop
		//----------------------------------------------------------------------

		// The film refresh time target
		const double targetTime = 0.2; // 200ms

        const u_int numSpatialReuseIterations = engine->numSpatialReuseIterations;

		u_int iterations = 4;

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
			// This is required for updating film denoiser parameter
			if (threadFilms[0]->film->GetDenoiser().IsEnabled()) {
				boost::unique_lock<boost::mutex> lock(engine->setKernelArgsMutex);
				SetAllAdvancePathsKernelArgs(0);
			}

            SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Beginning rendering for frame " << numFrames << ".");

			const double timeKernelStart = WallClockTime();

			// Get next sample if this is not the first iteration of this loop.
	        intersectionDevice->EnqueueKernel(advancePathsKernel_MK_NEXT_SAMPLE,
			    HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));

            // Generate camera rays for each pixel in this frame.
            intersectionDevice->EnqueueKernel(advancePathsKernel_MK_GENERATE_CAMERA_RAY,
			    HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));

            // Perform initial path resampling to get canonical samples for each pixel this frame.
            bool isInitialPathResamplingDone = false;
			bool firstLoop = false;
            u_int totalIterationsThisFrame = 0;

			SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Generating canonical initial path samples: " << taskCount);
			while (!isInitialPathResamplingDone) {
				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Queuing advance paths kernels for " << iterations << " iterations");

                // Trace until all paths are completed for this frame.
                for (u_int i = 0; i < iterations; ++i) {
                    // Trace rays
                    intersectionDevice->EnqueueTraceRayBuffer(raysBuff, hitsBuff, taskCount);

                    // Advance to next path state
                    EnqueueAdvancePathsKernel();
                }

				//SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Finished queuing advance paths kernels");

				//SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] All advance paths kernels finished execution");

                // Check if initial path resampling for all pixels is complete
				// This is blocking and waits for queue to finish
                isInitialPathResamplingDone = CheckSyncedPathStates(tasksStateReadBuffer, taskCount, slg::ocl::pathoclbase::PathState::SYNC);

                totalIterationsThisFrame += iterations;

				if (isInitialPathResamplingDone) {
					break;
				}
				firstLoop = false;
				iterations++;
            }

			SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Initial path resampling is complete, performing spatial reuse");
            
            // Perform spatial reuse.
			if (numSpatialReuseIterations > 0) {
				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Spatial reuse is enabled, performing " << numSpatialReuseIterations << " iterations");
				// Initialize variables and find spatial neighbors
				intersectionDevice->EnqueueKernel(spatialReuseInitKernel,
						HardwareDeviceRange(engine->taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
				// Ensure all paths are synced before continuing
				intersectionDevice->FinishQueue();
			} else {
				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Spatial reuse is disabled, configure with respirpathocl.spatialreuse.numiterations");
			}

			// Iterate x times
			for (u_int i = 0; i < numSpatialReuseIterations; i++) {
				SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Beginning spatial reuse iteration " << i);
				bool isSpatialReuseDone = false;
				// Resample neighboring pixels
				while (!isSpatialReuseDone) {
					// Resample next neighboring pixel
					intersectionDevice->EnqueueKernel(spatialReuseResampleNeighborKernel,
						HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
					
					SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Spatial reuse (iteration " << i << "): tracing shadow paths");
					// Evaluate visibility if resampling succeeds
					for (u_int i = 0; i < totalIterationsThisFrame; ++i) {
						// Trace shadow rays to reconnection vertices
						intersectionDevice->EnqueueTraceRayBuffer(raysBuff, hitsBuff, taskCount);

						// Check visibility and update reservoirs if successful
						intersectionDevice->EnqueueKernel(spatialReuseCheckVisibilityKernel,
							HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
					}

					// This is blocking and waits for queue to finish
					isSpatialReuseDone = CheckSyncedPathStates(tasksStateReadBuffer, taskCount, slg::ocl::pathoclbase::PathState::SYNC);
				}
				
				// Finish the iteration
				intersectionDevice->EnqueueKernel(spatialReuseFinishIterationKernel,
					HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));

				// Ensure all paths are synced before continuing
				intersectionDevice->FinishQueue();
			}
			
			SLG_LOG("[PathOCLRespirOCLRenderThread::" << threadIndex << "] Spatial reuse passes are complete, splatting pixels");

            // Splat pixels.
			intersectionDevice->EnqueueKernel(spatialReuseSetSplatKernel,
                    HardwareDeviceRange(engine->taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
            intersectionDevice->EnqueueKernel(advancePathsKernel_MK_SPLAT_SAMPLE,
			    HardwareDeviceRange(taskCount), HardwareDeviceRange(advancePathsWorkGroupSize));
			
			const double timeKernelEnd = WallClockTime();
			totalKernelTime += timeKernelEnd - timeKernelStart;

			//------------------------------------------------------------------

			const double timeTransferStart = WallClockTime();

			// Transfer the film from GPU to CPU only if I have already spent enough time running
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

            numFrames++;

			/*if (threadIndex == 0)
				SLG_LOG("[DEBUG] transfer time: " << (timeTransferEnd - timeTransferStart) * 1000.0 << "ms "
						"kernel time: " << (timeKernelEnd - timeKernelStart) * 1000.0 << "ms "
						"iterations: " << iterations << " #"<< taskCount << ")");*/

			// Check halt conditions
			if (engine->film->GetConvergence() == 1.f)
				break;
		}
        free(tasksStateReadBuffer);

	} catch (boost::thread_interrupted) {
		SLG_LOG("[PathOCLRenderThread::" << threadIndex << "] Rendering thread halted");
	}
    SLG_LOG("[PathOCLRespirRenderThread::" << threadIndex << "]: Rendered " << numFrames << " frames in total.");

	threadFilms[0]->RecvFilm(intersectionDevice);
	intersectionDevice->FinishQueue();
	
	threadDone = true;
	
	intersectionDevice->PopThreadCurrentDevice();
}

#endif
