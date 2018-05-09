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

void seq_seg::run(uchar* d, int width, int height, float t_split, float t_merge)
{
	data = d;
	image_w = width;
	image_h = height;
	threshold_split = t_split;
	threshold_merge = t_merge;
	cout << "segmentation running!\n";
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
	auto start_rag = high_resolution_clock::now();
	char* RAG = createRAG();
	auto end_rag = high_resolution_clock::now();
	duration<double> rag_time = end_rag - start_rag;
	cout << "RAG creation time " << rag_time.count() << "s\n";
	auto start_merge = high_resolution_clock::now();
	merge(RAG);
	auto end_merge = high_resolution_clock::now();
	duration<double> merge_time = end_merge - start_merge;
	cout << "merge time " << merge_time.count() << "s\n";;
	cout << "regions size after merging is " << getRegionsNumber() << "\n";
}

void seq_seg::displaySeg(uchar* d)
{
	for (int i = 0; i < regions.size; i++)
	{
		seq_seg::displayRegion(d, i);
	}
}

void seq_seg::getRegions(Regions &rs) const
{
	rs = regions;
}

void seq_seg::displayRegion(uchar* d, int reg_index)
{
	int start = regions.data[reg_index].index;
	int rWidth = regions.data[reg_index].width;
	int rHeight = regions.data[reg_index].height;
	int id = regions.data[reg_index].id;

	// ensure finding the right mean
	while (regions.data[id].id != id)
	{
		id = regions.data[id].id;
	}
	uchar mean = (uchar)regions.data[id].mean;

	for (int i = start; i < start + (image_w * rHeight); i += image_w)
	{
		for (int j = i; j < i + rWidth; j++)
		{
			d[j] = mean;
		}
	}
}

int seq_seg::getX(int index)
{
	return index % image_w;
}

int seq_seg::getY(int index)
{
	return index / image_w;
}

bool seq_seg::neighbors(Region &r1, Region &r2) {
	int x1 = getX(r1.index);
	int x2 = getX(r2.index);
	int y1 = getY(r1.index);
	int y2 = getY(r2.index);
	int width1 = r1.width;
	int width2 = r2.width;
	int height1 = r1.height;
	int height2 = r2.height;
	return !(x1 + width1 < x2
			|| x1 > x2 + width2
			|| y1 + height1 < y2
			|| y1 > y2 + height2);
}

void seq_seg::split(int start, int rWidth, int rHeight, vector<Region> &rs) 
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

float seq_seg::computeMean(int start, int rWidth, int rHeight) 
{
	return (float)computeSum(start, rWidth, rHeight) / ((float)(rWidth * rHeight));
}

int seq_seg::computeSum(int start, int rWidth, int rHeight)
{
	int sum = 0;
	for (int i = start; i < start + image_w * rHeight; i += image_w) {
		for (int j = i; j < i + rWidth; j++) {
			sum += data[j];
		}
	}
	return sum;
}

float seq_seg::computeSD(float mean, int start, int rWidth, int rHeight)
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

char* seq_seg::createRAG() 
{
	int n = regions.size * regions.size;
	char* RAG = new char[n];
	for (int i = 0; i < n; i++)
	{
		int row = i / regions.size;
		int col = i % regions.size;
		RAG[i] = neighbors(regions.data[row], regions.data[col]);
	}
	return RAG;
}

void seq_seg::merge(char* RAG) {
	for (int i = 0; i < regions.size; i++) {
		if (regions.data[i].id == i) {
			merge(i, RAG);
		}
	}
}

void seq_seg::merge(int curr, char* RAG) {
	for (int i = curr + 1; i < regions.size; i++)
	{
		if (regions.data[i].id == i && curr != i && shouldMerge(regions.data[curr], regions.data[i], RAG))
		{
			mergeRegions(RAG, regions.data[curr], regions.data[i]);
			i = -1;
		}
	}
}

void seq_seg::mergeRegions(char* RAG, Region &r1, Region &r2) {
	updateRAG(RAG, r1.id, r2.id);
	int tn = r1.n + r2.n;
	float mean = ((float)r1.n * r1.mean + (float)r2.n * r2.mean) / (float)tn;
	r1.mean, r2.mean = mean;
	r1.n, r2.n = tn;
	r2.id = r1.id;
}

void seq_seg::updateRAG(char* RAG, int x, int y)
{
	for (int i = 0; i < regions.size; i++)
	{
		RAG[x * regions.size + i] |= RAG[y * regions.size + i];
	}
}

bool seq_seg::shouldMerge(Region &curr, Region &candidate, char* RAG) {
	return RAG[curr.id * regions.size + candidate.id] && abs(curr.mean - candidate.mean) <= threshold_merge;
}

void seq_seg::initID() {
	for (int i = 0; i < regions.size; i++) {
		regions.data[i].id = i;
	}
}

int seq_seg::getRegionsNumber() {
	int n = 0;
	for (int i = 0; i < regions.size; i++) {
		if (regions.data[i].id == i) {
			n++;
		}
	}
	return n;
}