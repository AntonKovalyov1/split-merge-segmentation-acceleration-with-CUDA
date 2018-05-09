#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <chrono>

#include "RegSeg.h"


using namespace std;
using namespace chrono;

__global__
void updateRAGCUDA(char* RAG, int rs_n, int x, int y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < rs_n; i += stride)
	{
		RAG[x * rs_n + i] |= RAG[y * rs_n + i];
	}
}

__global__
void mergeReduction(char* RAG, Region* rs, int rs_n, int sample, int offset, float threshold, int n, int m)
{
	//current region id
	int index = (blockIdx.x * blockDim.x + threadIdx.x) * sample + offset;
	//comparison region id
	int comp_index = index + (sample / 2);
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < rs_n; i += stride)
	{
		Region &r1 = rs[index];
		Region &r2 = rs[comp_index];
		//check if regions are neighbors and homogeneity property is satisfied
		if (r1.id == index 
				&& r2.id == comp_index 
				&& RAG[index * rs_n + comp_index] 
				&& fabsf(r1.mean - r2.mean) < threshold)
		{
			//merge: recompute mean and size of merged region
			int tn = r1.n + r2.n;
			float mean = ((float)r1.n * r1.mean + (float)r2.n * r2.mean) / (float)tn;
			r1.mean, r2.mean = mean;
			r1.n, r2.n = tn;
			r2.id = r1.id;
			//update RAG in parallel in a separate kernel launch
			updateRAGCUDA<<<n, m>>>(RAG, rs_n, r1.id, r2.id);
		}
	}
}

__device__
char neighbors(Region &r1, Region &r2, int width)
{
	int x1 = r1.index % width;
	int x2 = r2.index % width;
	int y1 = r1.index / width;
	int y2 = r2.index / width;
	int width1 = r1.width;
	int width2 = r2.width;
	int height1 = r1.height;
	int height2 = r2.height;
	return !(x1 + width1 < x2
		|| x1 > x2 + width2
		|| y1 + height1 < y2
		|| y1 > y2 + height2
		|| r1.id == r2.id);
}

__global__
void computeRAG(Region curr, Region* rs, int rs_n, char* RAG, int image_w)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < rs_n; i += stride)
	{
		RAG[curr.id * rs_n + i] = neighbors(curr, rs[i], image_w);
	}
}

void cuda_seg::run(uchar* d, int width, int height, float t_split, float t_merge, bool reduction)
{
	data = d;
	image_w = width;
	image_h = height;
	threshold_split = t_split;
	threshold_merge = t_merge;

	//split
	auto start_split = high_resolution_clock::now();
	vector<Region> rs;
	split(0, image_w, image_h, rs);
	regions.size = (int)rs.size();
	regions.data = new Region[regions.size];
	copy(rs.begin(), rs.end(), regions.data);
	initID();
	auto end_split = high_resolution_clock::now();
	duration<double> split_time = end_split - start_split;
	cout << "split time " << split_time.count() << "s\n";
	cout << "region size after splitting is " << getRegionsNumber() << "\n";

	//create RAG
	createRAG();

    //merge
	auto start_merge = high_resolution_clock::now();
	if (reduction)
	{
		reduce();
	}
	merge(RAG);
	auto end_merge = high_resolution_clock::now();
	duration<double> merge_time = end_merge - start_merge;
	cout << "merge time " << merge_time.count() << "s\n";;
	cout << "regions size after merging is " << getRegionsNumber() << "\n";
}

void cuda_seg::displaySeg(uchar* d)
{
	for (int i = 0; i < regions.size; i++)
	{
		cuda_seg::displayRegion(d, i);
	}
}

void cuda_seg::getRegions(Regions &rs) const
{
	rs = regions;
}

void cuda_seg::displayRegion(uchar* d, int reg_index)
{
	int start = regions.data[reg_index].index;
	int rWidth = regions.data[reg_index].width;
	int rHeight = regions.data[reg_index].height;
	int id = regions.data[reg_index].id;

	// ensure finding the right mean
	id = regions.data[id].id;
	uchar mean = (uchar)regions.data[id].mean;

	for (int i = start; i < start + (image_w * rHeight); i += image_w)
	{
		for (int j = i; j < i + rWidth; j++)
		{
			d[j] = mean;
		}
	}
}

void cuda_seg::split(int start, int rWidth, int rHeight, vector<Region> &rs)
{
	float mean = computeMean(start, rWidth, rHeight);
	if (rWidth <= 2 || rHeight <= 2)
	{
		Region r(start, rWidth, rHeight, rWidth * rHeight, mean);
		rs.push_back(r);
	}
	else if (computeSD(mean, start, rWidth, rHeight) <= threshold_split)
	{
		Region r(start, rWidth, rHeight, rWidth * rHeight, mean);
		rs.push_back(r);
	}
	else
	{
		int hrWidth = rWidth / 2;
		int hrHeight = rHeight / 2;
		split(start, hrWidth, hrHeight, rs);
		split(start + hrWidth, rWidth - hrWidth, hrHeight, rs);
		split(start + (image_w * hrHeight), hrWidth, rHeight - hrHeight, rs);
		split(start + (image_w * hrHeight) + hrWidth, rWidth - hrWidth, rHeight - hrHeight, rs);
	}
}

float cuda_seg::computeMean(int start, int rWidth, int rHeight)
{
	return (float)computeSum(start, rWidth, rHeight) / ((float)(rWidth * rHeight));
}

int cuda_seg::computeSum(int start, int rWidth, int rHeight)
{
	int sum = 0;
	for (int i = start; i < start + image_w * rHeight; i += image_w) {
		for (int j = i; j < i + rWidth; j++) {
			sum += data[j];
		}
	}
	return sum;
}

float cuda_seg::computeSD(float mean, int start, int rWidth, int rHeight)
{
	float sum = 0;
	for (int i = start; i < start + image_w * rHeight; i += image_w)
	{
		for (int j = i; j < i + rWidth; j++)
		{
			sum += pow(mean - data[j], 2);
		}
	}
	return (float)pow(sum / (float)(rWidth * rHeight), 0.5);
}

void cuda_seg::createRAG()
{
	int n = regions.size * regions.size;
	RAG = new char[n];

	// allocate memory in device
	cudaMalloc((void **)&d_RAG, n * sizeof(char));
	cudaMalloc((void **)&d_rs, regions.size * sizeof(Region));

	// copy regions to device regions
	cudaMemcpy(d_rs, regions.data, regions.size * sizeof(Region), cudaMemcpyHostToDevice);

	//compute RAG
	int blockSize = 128;
	int blocksNum = (regions.size + blockSize - 1) / blockSize;
	auto s = high_resolution_clock::now();
	for (int i = 0; i < regions.size; i++)
	{
		computeRAG << <blocksNum, blockSize >> >(regions.data[i], d_rs, regions.size, d_RAG, image_w);
	}
	cudaError_t err = cudaDeviceSynchronize();
	cout << "CUDA error: " << cudaGetErrorString(err) << "\n";
	auto e = high_resolution_clock::now();
	duration<double> time = e - s;
	cout << "RAG kernel creation time: " << time.count() << "\n";

	// copy results to host
	cudaMemcpy(RAG, d_RAG, n * sizeof(char), cudaMemcpyDeviceToHost);
}

void cuda_seg::reduce()
{
	int blockSize = 128;
	int blocksNum = (regions.size + blockSize - 1) / blockSize;
	int blockSizeMerge = 32;
	int sample = 2;
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j <= i; j++)
		{
			mergeReduction << <regions.size / (sample * blockSizeMerge), blockSizeMerge >> > (
				d_RAG, d_rs, regions.size, sample, j, threshold_merge, blocksNum, blockSize);
		}
		cudaDeviceSynchronize();
		sample += 2;
	}

	// copy results to host
	cudaMemcpy(regions.data, d_rs, regions.size * sizeof(Region), cudaMemcpyDeviceToHost);
	cudaMemcpy(RAG, d_RAG, regions.size * regions.size * sizeof(char), cudaMemcpyDeviceToHost);

	cout << "regions size after reduction " << getRegionsNumber() << "\n";
}

void cuda_seg::merge(char* RAG) 
{
	for (int i = 0; i < regions.size; i++) 
	{
		if (regions.data[i].id == i) 
		{
			merge(i, RAG);
		}
	}
}

void cuda_seg::merge(int curr, char* RAG) 
{
	for (int i = curr + 1; i < regions.size; i++)
	{
		if (regions.data[i].id == i && i != curr && shouldMerge(regions.data[curr], regions.data[i], RAG))
		{
			mergeRegions(RAG, regions.data[curr], regions.data[i]);
			i = -1;
		}
	}
}

void cuda_seg::mergeRegions(char* RAG, Region &r1, Region &r2) 
{
	updateRAG(RAG, r1.id, r2.id);
	int tn = r1.n + r2.n;
	float mean = ((float)r1.n * r1.mean + (float)r2.n * r2.mean) / (float)tn;
	r1.mean, r2.mean = mean;
	r1.n, r2.n = tn;
	r2.id = r1.id;
}

void cuda_seg::updateRAG(char* RAG, int x, int y)
{
	for (int i = 0; i < regions.size; i++)
	{
		RAG[x * regions.size + i] |= RAG[y * regions.size + i];
	}
}

bool cuda_seg::shouldMerge(Region &curr, Region &candidate, char* RAG) 
{
	return RAG[curr.id * regions.size + candidate.id] && abs(curr.mean - candidate.mean) <= threshold_merge;
}

void cuda_seg::initID() 
{
	for (int i = 0; i < regions.size; i++) 
	{
		regions.data[i].id = i;
	}
}

int cuda_seg::getRegionsNumber() 
{
	int n = 0;
	for (int i = 0; i < regions.size; i++) 
	{
		if (regions.data[i].id == i) 
		{
			n++;
		}
	}
	return n;
}