#include "morphology.h"


template <typename T>
__global__ void ErodeSharedStep1(const int batch_size, const T *src, T *dst, int radio, int width, int height, int tile_w, int tile_h)
{
	extern __shared__ int smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx * tile_w + tx - radio;
	int y = by * tile_h + ty;
	
	smem[ty * blockDim.x + tx] = 255;
	__syncthreads();
	if( x < 0 || x >= width || y >= height * batch_size) {
		return;
	}
	smem[ty * blockDim.x + tx] = (int)src[y * width + x];
	__syncthreads();

	if( x < (bx * tile_w) || x >= (bx + 1) * tile_w ) {
		return;
	}
	
	int *smem_thread = &smem[ty * blockDim.x + tx - radio];
	int val = smem_thread[0];
	for( int i = 1; i <= 2 * radio; i++ ) {
		val = MIN( val, smem_thread[i] );
	}
	dst[y * width + x] = (uint8_t)val;
}

template <typename T>
__global__ void ErodeSharedStep2(const int batch_size, const T *src, T *dst, int radio, int width, int height, int tile_w, int tile_h)
{
	extern __shared__ int smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx * tile_w + tx;
	int y = by * tile_h + ty - radio;

	smem[ty * blockDim.x + tx] = 255;
	__syncthreads();
	if( x >= width || y < 0 || y >= height * batch_size) {
		return;
	}
	smem[ty * blockDim.x + tx] = (int)src[y * width + x];
	__syncthreads();

	if( y < (by * tile_h) || y >= (by + 1) * tile_h) {
		return;
	}

	int *smem_thread = &smem[(ty - radio) * blockDim.x + tx];
	int val = smem_thread[0];
	for( int i = 1; i <= 2 * radio; i++ ) {
		val = MIN( val, smem_thread[i * blockDim.x] );
	}
	dst[y * width + x] = (uint8_t)val;
}

template <typename T>
__global__ void ErodeSharedStep1(const T *src, T *dst, int radio, int width, int height, int tile_w, int tile_h)
{
	extern __shared__ int smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx * tile_w + tx - radio;
	int y = by * tile_h + ty;
	
	smem[ty * blockDim.x + tx] = 255;
	__syncthreads();
	if( x < 0 || x >= width || y >= height) {
		return;
	}
	smem[ty * blockDim.x + tx] = (int)src[y * width + x];
	__syncthreads();

	if( x < (bx * tile_w) || x >= (bx + 1) * tile_w ) {
		return;
	}
	
	int *smem_thread = &smem[ty * blockDim.x + tx - radio];
	int val = smem_thread[0];
	for( int i = 1; i <= 2 * radio; i++ ) {
		val = MIN( val, smem_thread[i] );
	}
	dst[y * width + x] = (uint8_t)val;
}

template <typename T>
__global__ void ErodeSharedStep2(const T *src, T *dst, int radio, int width, int height, int tile_w, int tile_h)
{
	extern __shared__ int smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx * tile_w + tx;
	int y = by * tile_h + ty - radio;

	smem[ty * blockDim.x + tx] = 255;
	__syncthreads();
	if( x >= width || y < 0 || y >= height) {
		return;
	}
	smem[ty * blockDim.x + tx] = (int)src[y * width + x];
	__syncthreads();

	if( y < (by * tile_h) || y >= (by + 1) * tile_h ) {
		return;
	}

	int *smem_thread = &smem[(ty - radio) * blockDim.x + tx];
	int val = smem_thread[0];
	for( int i = 1; i <= 2 * radio; i++ ) {
		val = MIN( val, smem_thread[i * blockDim.x] );
	}
	dst[y * width + x] = (uint8_t)val;
}

template <typename T>
__global__ void DilateSharedStep1(const T *src, T *dst, int radio, int width, int height, int tile_w, int tile_h)
{
	extern __shared__ int smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx * tile_w + tx - radio;
	int y = by * tile_h + ty;
	
	smem[ty * blockDim.x + tx] = 0;
	__syncthreads();
	if( x < 0 || x >= width || y >= height ) {
		return;
	}
	smem[ty * blockDim.x + tx] = (int)src[y * width + x];
	__syncthreads();

	if( x < (bx * tile_w) || x >= (bx + 1) * tile_w ) {
		return;
	}
	
	int *smem_thread = &smem[ty * blockDim.x + tx - radio];
	int val = smem_thread[0];
	for( int i = 1; i <= 2 * radio; i++ ) {
		val = MAX( val, smem_thread[i] );
	}
	dst[y * width + x] = (uint8_t)val;
}

template <typename T>
__global__ void DilateSharedStep2(const T *src, T *dst, int radio, int width, int height, int tile_w, int tile_h)
{
	extern __shared__ int smem[];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int x = bx * tile_w + tx;
	int y = by * tile_h + ty - radio;

	smem[ty * blockDim.x + tx] = 0;
	__syncthreads();
	if( x >= width || y < 0 || y >= height ) {
		return;
	}
	smem[ty * blockDim.x + tx] = (int)src[y * width + x];
	__syncthreads();

	if( y < (by * tile_h) || y >= (by + 1) * tile_h ) {
		return;
	}

	int *smem_thread = &smem[(ty - radio) * blockDim.x + tx];
	int val = smem_thread[0];
	for( int i = 1; i <= 2 * radio; i++ ) {
		val = MAX( val, smem_thread[i * blockDim.x] );
	}
	dst[y * width + x] = (uint8_t)val;
}

void ErodeTwoStepShared(const int batch_size, void *src, void *temp, void *dst, 
    int radio, int width, int height, cudaStream_t& stream)
{	
	int tile_w1 = 640;
	int tile_h1 = 1;
	dim3 block1( tile_w1 + 2 * radio, tile_h1 );
	dim3 grid1( ceil((float)width / tile_w1), ceil(batch_size * (float)height / tile_h1) );
    ErodeSharedStep1<<<grid1, block1, block1.y * block1.x * sizeof(int), stream>>>(
		batch_size, 
        static_cast<const uint8_t* const>(src), 
        static_cast<uint8_t*>(temp), 
        radio, width, height, tile_w1, tile_h1);
	// cudaDeviceSynchronize();

	int tile_w2 = 8;
	int tile_h2 = 64;
	dim3 block2( tile_w2, tile_h2 + 2 * radio );
	dim3 grid2( ceil((float)width / tile_w2), ceil(batch_size * (float)height / tile_h2) );
	ErodeSharedStep2<<<grid2, block2, block2.y * block2.x * sizeof(int), stream>>>(
		batch_size, 
        static_cast<const uint8_t* const>(temp), 
        static_cast<uint8_t*>(dst), 
        radio, width, height, tile_w2, tile_h2);
	// cudaDeviceSynchronize();
}

void ErodeTwoStepShared(void *src, void *temp, void *dst, 
    int radio, int width, int height, cudaStream_t& stream)
{	
	int tile_w1 = 640;
	int tile_h1 = 1;
	dim3 block1( tile_w1 + 2 * radio, tile_h1 );
	dim3 grid1( ceil((float)width / tile_w1), ceil((float)height / tile_h1) );
    ErodeSharedStep1<<<grid1, block1, block1.y * block1.x * sizeof(int), stream>>>(
        static_cast<const uint8_t* const>(src), 
        static_cast<uint8_t*>(temp), 
        radio, width, height, tile_w1, tile_h1);
	// cudaDeviceSynchronize();

	int tile_w2 = 8;
	int tile_h2 = 64;
	dim3 block2( tile_w2, tile_h2 + 2 * radio );
	dim3 grid2( ceil((float)width / tile_w2), ceil((float)height / tile_h2) );
	ErodeSharedStep2<<<grid2, block2, block2.y * block2.x * sizeof(int), stream>>>(
        static_cast<const uint8_t* const>(temp), 
        static_cast<uint8_t*>(dst), 
        radio, width, height, tile_w2, tile_h2);
	// cudaDeviceSynchronize();
}

void ErodeTwoStepShared(void *src, void *dst, 
    int radio, int width, int height, cudaStream_t& stream)
{
	void *temp = NULL;
	cudaMalloc( &temp, width * height * sizeof(uint8_t) );
	
	int tile_w1 = 640;
	int tile_h1 = 1;
	dim3 block1( tile_w1 + 2 * radio, tile_h1 );
	dim3 grid1( ceil((float)width / tile_w1), ceil((float)height / tile_h1) );
    ErodeSharedStep1<<<grid1, block1, block1.y * block1.x * sizeof(int), stream>>>(
        static_cast<const uint8_t* const>(src), 
        static_cast<uint8_t*>(temp), 
        radio, width, height, tile_w1, tile_h1);
	// cudaDeviceSynchronize();

	int tile_w2 = 8;
	int tile_h2 = 64;
	dim3 block2( tile_w2, tile_h2 + 2 * radio );
	dim3 grid2( ceil((float)width / tile_w2), ceil((float)height / tile_h2) );
	ErodeSharedStep2<<<grid2, block2, block2.y * block2.x * sizeof(int), stream>>>(
        static_cast<const uint8_t* const>(temp), 
        static_cast<uint8_t*>(dst), 
        radio, width, height, tile_w2, tile_h2);
	// cudaDeviceSynchronize();

	cudaFree( temp );
}

void DilateTwoStepShared(void *src, void *temp, void *dst, 
    int radio, int width, int height, cudaStream_t& stream)
{	
	int tile_w1 = 640;
	int tile_h1 = 1;
	dim3 block1( tile_w1 + 2 * radio, tile_h1 );
	dim3 grid1( ceil((float)width / tile_w1), ceil((float)height / tile_h1) );
	DilateSharedStep1<<<grid1, block1, block1.y * block1.x * sizeof(int), stream>>>(
        static_cast<const uint8_t* const>(src), 
        static_cast<uint8_t*>(temp), 
        radio, width, height, tile_w1, tile_h1);
	// cudaDeviceSynchronize();

	int tile_w2 = 8;
	int tile_h2 = 64;
	dim3 block2( tile_w2, tile_h2 + 2 * radio );
	dim3 grid2( ceil((float)width / tile_w2), ceil((float)height / tile_h2) );
	DilateSharedStep2<<<grid2, block2, block2.y * block2.x * sizeof(int), stream>>>(
        static_cast<const uint8_t* const>(temp), 
        static_cast<uint8_t*>(dst), 
        radio, width, height, tile_w2, tile_h2);
	// cudaDeviceSynchronize();
}

void DilateTwoStepShared(void *src, void *dst, 
    int radio, int width, int height, cudaStream_t& stream)
{
	void *temp = NULL;
	cudaMalloc( &temp, width * height * sizeof(uint8_t) );
	
	int tile_w1 = 640;
	int tile_h1 = 1;
	dim3 block1( tile_w1 + 2 * radio, tile_h1 );
	dim3 grid1( ceil((float)width / tile_w1), ceil((float)height / tile_h1) );
	DilateSharedStep1<<<grid1, block1, block1.y * block1.x * sizeof(int), stream>>>(
        static_cast<const uint8_t* const>(src), 
        static_cast<uint8_t*>(temp), 
        radio, width, height, tile_w1, tile_h1);
	// cudaDeviceSynchronize();

	int tile_w2 = 8;
	int tile_h2 = 64;
	dim3 block2( tile_w2, tile_h2 + 2 * radio );
	dim3 grid2( ceil((float)width / tile_w2), ceil((float)height / tile_h2) );
	DilateSharedStep2<<<grid2, block2, block2.y * block2.x * sizeof(int), stream>>>(
        static_cast<const uint8_t* const>(temp), 
        static_cast<uint8_t*>(dst), 
        radio, width, height, tile_w2, tile_h2);
	// cudaDeviceSynchronize();

	cudaFree( temp );
}
