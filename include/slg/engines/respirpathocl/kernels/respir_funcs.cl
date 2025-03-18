#line 2 "respir_funcs.cl"

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

 #ifndef DEBUG_GID
 #define DEBUG_GID 159982
 #endif

/*
// Radiance group utility functions
*/
OPENCL_FORCE_INLINE float3 SampleResult_GetUnscaledSpectrum(__constant const Film* restrict film,
    const SampleResult *sampleResult) {
    float3 c = BLACK;

    for (uint i = 0; i < film->radianceGroupCount; ++i)
        c += VLOAD3F(sampleResult->radiancePerPixelNormalized[i].c);

    return c;
}

OPENCL_FORCE_INLINE float Radiance_Y(__constant const Film* restrict film,
    const Spectrum *radianceGroups) {
    float y = 0.f;

    for (uint i = 0; i < film->radianceGroupCount; ++i)
        y += Spectrum_Y(VLOAD3F(radianceGroups[i].c));

    return y;
}

OPENCL_FORCE_INLINE float Radiance_Filter(__constant const Film* restrict film,
    const Spectrum *radianceGroups) {
    float f = 0.f;

    for (uint i = 0; i < film->radianceGroupCount; ++i)
        f += Spectrum_Filter(VLOAD3F(radianceGroups[i].c));

    return f;
}

OPENCL_FORCE_INLINE bool Radiance_IsBlack(__constant const Film* restrict film,
const Spectrum *radianceGroups) {
    for (uint i = 0; i < film->radianceGroupCount; ++i) {
        if (!Spectrum_IsBlack(VLOAD3F(radianceGroups[i].c))) {
            return false;
        }
    }
    return true;
}

OPENCL_FORCE_INLINE float3 Radiance_GetUnscaledSpectrum(__constant const Film* restrict film,
    const Spectrum *radianceGroups) {
    float3 c = BLACK;

    for (uint i = 0; i < film->radianceGroupCount; ++i)
        c += VLOAD3F(radianceGroups[i].c);

    return c;
}

OPENCL_FORCE_INLINE void Radiance_Clear(Spectrum* radiance) {
    VSTORE3F(BLACK, radiance[0].c);
    VSTORE3F(BLACK, radiance[1].c);
    VSTORE3F(BLACK, radiance[2].c);
    VSTORE3F(BLACK, radiance[3].c);
    VSTORE3F(BLACK, radiance[4].c);
    VSTORE3F(BLACK, radiance[5].c);
    VSTORE3F(BLACK, radiance[6].c);
    VSTORE3F(BLACK, radiance[7].c);
}

OPENCL_FORCE_INLINE void Radiance_Add(__constant const Film* restrict film, 
    const Spectrum* a, const Spectrum* b, Spectrum* out) {
    for (uint i = 0; i < film->radianceGroupCount; i++) {
        VSTORE3F(VLOAD3F(a[i].c) + VLOAD3F(b[i].c), out[i].c);
    }
}

OPENCL_FORCE_INLINE void Radiance_Add_Weighted(__constant const Film* restrict film, 
    const Spectrum* a, const Spectrum* b, const float weight, Spectrum* out) {
    for (uint i = 0; i < film->radianceGroupCount; i++) {
        VSTORE3F((VLOAD3F(a[i].c) + (VLOAD3F(b[i].c) * weight)), out[i].c);
    }
}

OPENCL_FORCE_INLINE void Radiance_Scale(__constant const Film* restrict film, 
    const Spectrum* a, const float scale, Spectrum* out) {
    for (uint i = 0; i < film->radianceGroupCount; i++) {
        VSTORE3F((VLOAD3F(a[i].c) * scale), out[i].c);
    }
}

OPENCL_FORCE_INLINE void Radiance_ScaleGroup(__constant const Film* restrict film, 
    const Spectrum* a, const float3 scale, Spectrum* out) {
    for (uint i = 0; i < film->radianceGroupCount; i++) {
        VSTORE3F((VLOAD3F(a[i].c) * scale), out[i].c);
    }
}

OPENCL_FORCE_INLINE void Radiance_Sub(__constant const Film* restrict film, 
    const Spectrum* a, const Spectrum* b, Spectrum* out) {
    for (uint i = 0; i < film->radianceGroupCount; i++) {
        VSTORE3F(VLOAD3F(a[i].c) - VLOAD3F(b[i].c), out[i].c);
    }
}

OPENCL_FORCE_INLINE void Radiance_Copy(__constant const Film* restrict film, 
    const Spectrum* radiance, Spectrum* out) {
    for (uint i = 0; i < film->radianceGroupCount; i++) {
        VSTORE3F(VLOAD3F(radiance[i].c), out[i].c);
    }
}

/*
// Respir init functions
*/
OPENCL_FORCE_INLINE void RespirReservoir_Init(RespirReservoir* restrict reservoir) {
	reservoir->weight = 0.0f;
	reservoir->M = 0.0f;
	Radiance_Clear(reservoir->sample.integrand);
	VSTORE3F(BLACK, &reservoir->sample.prefixBsdf.hitPoint.p.x);
	reservoir->sample.lightPdf = 0.0f;
	reservoir->sample.hitTime = 0.0f;
	reservoir->sample.pathDepth = -1;
	reservoir->sample.isLastVertexNee = false;
	Radiance_Clear(reservoir->sample.rc.irradiance);
	VSTORE3F(BLACK, &reservoir->sample.rc.bsdf.hitPoint.p.x);
	VSTORE3F(WHITE, &reservoir->sample.rc.incidentDir.x);
	VSTORE3F(WHITE, reservoir->sample.rc.incidentBsdfValue.c);
	reservoir->sample.rc.incidentPdf = 1.0f;
	reservoir->sample.rc.prefixToRcPdf = 1.0f;
	reservoir->sample.rc.jacobian = 1.0f;
	reservoir->sample.rc.pathDepth = -1;
}

OPENCL_FORCE_INLINE void Respir_Init(GPUTaskState* restrict taskState) {
    RespirReservoir_Init(&taskState->reservoir);

	VSTORE3F(WHITE, taskState->currentThroughput.c);
	VSTORE3F(WHITE, taskState->pathPdf.c);
	taskState->rrProbProd = 1.0f;
	taskState->lastDirectLightMisWeight = 1.0f;
	VSTORE3F(WHITE, taskState->lastDirectLightBsdfEval.c);
}

/*
// Respir utility functions
*/
OPENCL_FORCE_INLINE int PixelIndexMap_Get(__global const int* pixelIndexMap, const uint mapWidth, const uint x, const uint y) {
	return pixelIndexMap[y * mapWidth + x];
}

OPENCL_FORCE_INLINE void PixelIndexMap_Set(__global int* pixelIndexMap, const uint mapWidth, const uint x, const uint y, const int value) {
	pixelIndexMap[y * mapWidth + x] = value;
}

// Add a sample to the streaming reservoir.
// Simply replace based on the new sample's weight and the reservoir's current sum weight.
OPENCL_FORCE_INLINE bool RespirReservoir_Add(RespirReservoir* restrict reservoir, 
	const Spectrum* restrict pathRadiance, float pdf, 
	Seed* restrict seed, __constant const Film* restrict film) 
{
	// increase sample count
	reservoir->M++;

	float weight = Radiance_Filter(film, pathRadiance) / pdf;

	// return if no chance of selection
	if (isinf(weight) || isnan(weight) || weight == 0.f) {
		return false;
	}

	reservoir->weight += weight;

	if (get_global_id(0) == DEBUG_GID) {
		printf("Initial path resampling: Resampling with weight %f against sum weight %f\n", weight, reservoir->weight);
	}

	if (Rnd_FloatValue(seed) * reservoir->weight <= weight) {
		Radiance_Copy(film, pathRadiance, reservoir->sample.integrand);
		return true;
	}
	return false;
}

// Merge inReservoir into outReservoir.
OPENCL_FORCE_INLINE bool RespirReservoir_Merge(RespirReservoir* restrict outReservoir, 
	const Spectrum* restrict inRadiance, const float inJacobian, const RespirReservoir* inReservoir,
	const float misWeight, Seed* restrict seed, __constant const Film* restrict film) 
{
	float weight = Radiance_Filter(film, inRadiance) * inJacobian * inReservoir->weight * misWeight;

	// Add sample counts
	outReservoir->M += inReservoir->M;

	// return if no chance of selection
	if (isinf(weight) || isnan(weight) || weight == 0.f) {
		return false;
	}

	outReservoir->weight += weight;
	if (get_global_id(0) == DEBUG_GID) {
		printf("Merging integrand (%f) with weight %f out of total weight %f.\n", 
				Radiance_Filter(film, inRadiance), weight, outReservoir->weight
		);
	}
	if (Rnd_FloatValue(seed) * outReservoir->weight <= weight) {
		outReservoir->sample = inReservoir->sample;
		Radiance_Copy(film, inRadiance, outReservoir->sample.integrand);
		return true;
	}
	return false;
}

OPENCL_FORCE_INLINE bool RespirReservoir_AddVertex(
		// incidentDir is the vector from the previous vertex to this one
		RespirReservoir* restrict reservoir, const float3 incidentDir,
		// integrand is the path contribution up to and including this vertex
		// postfixRadiance is the radiance emitted from the vertex
		const Spectrum* restrict integrand, const Spectrum* restrict postfixRadiance,
		// misWeight is the mis of the vertex's bsdfPdfW and lightPdf
		// lightPdf is the NEE light pick prob
		// pathPdf is cumulative product of all bsdfPdfW and connectionThroughput
		// rrProbProd is the cumulative product of the russian roulette probability so far
		const float misWeight, const float rrProbProd, const float lightPdf,
		const int pathDepth, Seed* restrict seed, __constant const Film* restrict film)
{
	// Resample for path integrand
	// Can't choose primary vertices
	if (pathDepth >= 1 && RespirReservoir_Add(reservoir, integrand, rrProbProd, seed, film)) {
		reservoir->sample.pathDepth = pathDepth;
		reservoir->sample.lightPdf = lightPdf;
		if (get_global_id(0) == DEBUG_GID) {
			printf("\tSelected new vertex at depth: %d\n", pathDepth);
		}
		Radiance_Copy(film, postfixRadiance, reservoir->sample.rc.irradiance);
		if (pathDepth == reservoir->sample.rc.pathDepth) {
			// cache reconnection vertex info
			VSTORE3F(incidentDir, &reservoir->sample.rc.incidentDir.x);
		}
		return true;
	}
	return false;
}

OPENCL_FORCE_INLINE bool RespirReservoir_AddNEEVertex(
		// incidentDir is the vector from the previous vertex to this one
		RespirReservoir* restrict reservoir, const float3 incidentDir,
		// integrand is the path contribution up to and including this vertex
		// postfixRadiance is the radiance emitted from the vertex
		const Spectrum* restrict integrand, const Spectrum* restrict postfixRadiance,
		// misWeight is the mis of the vertex's bsdfPdfW and lightPdf
		// lightPdf is the NEE light pick prob
		// pathPdf is cumulative product of all bsdfPdfW and connectionThroughput
		// rrProbProd is the cumulative product of the russian roulette probability so far
		const float misWeight, const float rrProbProd, const float lightPdf,
		const int pathDepth, Seed* restrict seed, __constant const Film* restrict film)
{
	if (get_global_id(0) == DEBUG_GID) {
		printf("Initial path resampling (NEE): Resampling with rr prob %f at depth %d\n", rrProbProd, pathDepth);
	}
	bool selected = RespirReservoir_AddVertex(reservoir, incidentDir, integrand, postfixRadiance, misWeight, rrProbProd, lightPdf, pathDepth, seed, film);
	if (selected && pathDepth == reservoir->sample.rc.pathDepth) {
		reservoir->sample.isLastVertexNee = true;
		Radiance_Scale(film, reservoir->sample.rc.irradiance, 
			lightPdf / misWeight, reservoir->sample.rc.irradiance);
	}
	return selected;
}

OPENCL_FORCE_INLINE bool RespirReservoir_AddEscapeVertex(
		// incidentDir is the vector from the previous vertex to this one
		RespirReservoir* restrict reservoir, const float3 incidentDir,
		// integrand is the path contribution up to and including this vertex
		// postfixRadiance is the radiance emitted from the vertex
		const Spectrum* restrict integrand, const Spectrum* restrict postfixRadiance,
		// misWeight is the mis of the vertex's bsdfPdfW and lightPdf
		// lightPdf is the NEE light pick prob
		// pathPdf is cumulative product of all bsdfPdfW and connectionThroughput
		// rrProbProd is the cumulative product of the russian roulette probability so far
		const float misWeight, const float rrProbProd, const float lightPdf,
		const int pathDepth, Seed* restrict seed, __constant const Film* restrict film)
{
	if (get_global_id(0) == DEBUG_GID) {
		printf("Initial path resampling (BSDF/Escape): Resampling with rr prob %f at depth %d\n", rrProbProd, pathDepth);
	}
	bool selected = RespirReservoir_AddVertex(reservoir, incidentDir, integrand, postfixRadiance, misWeight, rrProbProd, lightPdf, pathDepth, seed, film);
	if (selected && pathDepth == reservoir->sample.rc.pathDepth) {
		reservoir->sample.isLastVertexNee = false;
		Radiance_Scale(film, reservoir->sample.rc.irradiance, 
			1.0f / misWeight, reservoir->sample.rc.irradiance);
	}
	return selected;
}

// Set rcVertex info after BSDF scatter from rcVertex
OPENCL_FORCE_INLINE bool RespirReservoir_SetRcVertex(
	RespirReservoir* reservoir, const int pathDepth, const BSDF* bsdf, const float3 incidentDir, 
	const float incidentPdf, const float3 incidentBsdfValue, const float worldRadius
	MATERIALS_PARAM_DECL
) {
	if (get_global_id(0) == DEBUG_GID) {
		printf("Initial path resampling: Cached reconnection vertex info.\n");
	}
	VSTORE3F(incidentDir, &reservoir->sample.rc.incidentDir.x);
	reservoir->sample.rc.incidentPdf = incidentPdf;
	VSTORE3F(incidentBsdfValue, reservoir->sample.rc.incidentBsdfValue.c);
	
	const float3 toRc = VLOAD3F(&bsdf->hitPoint.p.x) - VLOAD3F(&reservoir->sample.prefixBsdf.hitPoint.p.x);
	const float distanceSquared = dot(toRc, toRc);
	const float cosAngle = abs(dot(VLOAD3F(&bsdf->hitPoint.fixedDir.x), BSDF_GetLandingGeometryN(bsdf)));
	// check roughness and distance connectability requirements
	// assume glossiness range is [0.f,1.f], and 1-glossiness is the roughness
	// distance threshold of 2-5% world size recommended by GRIS paper
	const float maxGlossiness = 0.2;
	const float minDistance = worldRadius * 2.0f * 0.025f;
	if (sqrt(distanceSquared) >= minDistance
		&& BSDF_GetGlossiness(&reservoir->sample.prefixBsdf MATERIALS_PARAM) <= maxGlossiness
		&& BSDF_GetGlossiness(bsdf MATERIALS_PARAM) <= maxGlossiness) {
		// cache partial jacobian here (squared distance / cos angle from rc norm)
		reservoir->sample.rc.jacobian = distanceSquared / cosAngle;
		reservoir->sample.rc.pathDepth = pathDepth;
		reservoir->sample.rc.bsdf = *bsdf;
		return true;
	} 
	
	if (get_global_id(0) == DEBUG_GID) {
		printf("Initial path resampling: Rejected reconnection vertex based on glossiness or distance.\n");
	}
	return false;
}

OPENCL_FORCE_INLINE void Respir_HandleInvalidShift(ShiftInOutData* shiftData,
		RespirReservoir* restrict out, PathState* restrict pathState) 
{
	out->sample.rc.jacobian = 0.0f;
	Radiance_Clear(out->sample.integrand);
	*pathState = (PathState) shiftData->afterShiftState;
	return;
}

OPENCL_FORCE_INLINE bool Respir_IsInvalidJacobian(const float jacobianDeterminant) {
	return jacobianDeterminant <= 0.0f || isnan(jacobianDeterminant) || isinf(jacobianDeterminant);
}

// Find the spatial neighbors around the pixel this work-item is handling for now
// Spatial radius is the square grid distance, not circle distance
// Advances taskState->neighborGid to the next neighbor.
// Return true if a neighbor was found, otherwise false.
// PRECONDITION: numNeighborsLeft > 0
// TODO: upgrade to n-rooks sampling around pixel and customizable spatial radius and number of spatial neighbors
OPENCL_FORCE_INLINE bool Respir_UpdateNextNeighborGid(SpatialReuseData* restrict srData, 
		const SampleResult* restrict sampleResult, const int spatialRadius,
		const int* restrict pixelIndexMap, const uint filmWidth, const uint filmHeight, Seed* restrict seed) 
{
	// DEBUG: IDENTITY SHIFT
	srData->neighborGid = get_global_id(0);
	srData->numNeighborsLeft--;
	return true;
	
	srData->neighborGid = -1;

	// randomly choose a pixel in the radius (inclusive) not including self
	int searchX = sampleResult->pixelX + (int) copysign(floor(Rnd_FloatValue(seed) * spatialRadius) + 1.0f, Rnd_FloatValue(seed) - 0.5f);
	int searchY = sampleResult->pixelY + (int) copysign(floor(Rnd_FloatValue(seed) * spatialRadius) + 1.0f, Rnd_FloatValue(seed) - 0.5f);
	srData->numNeighborsLeft--;

	if (get_global_id(0) == DEBUG_GID) {
		printf("Pixel (%d, %d), resampling (%d, %d).\n", sampleResult->pixelX, sampleResult->pixelY, searchX, searchY);
	}

	if (searchX >= 0 && searchX < filmWidth && searchY >= 0 && searchY < filmHeight // check in bounds
		&& (searchX != sampleResult->pixelX || searchY != sampleResult->pixelY)) // check not the pixel itself
	{ 
		srData->neighborGid = PixelIndexMap_Get(pixelIndexMap, filmWidth, searchX, searchY);
		// check that the neighbor is actually being worked on by a gputask
		if (srData->neighborGid != -1) {
			srData->numValidNeighbors++;
			// Successfully found valid neighbor
			return true;
		}
	}
	
	return false;
}