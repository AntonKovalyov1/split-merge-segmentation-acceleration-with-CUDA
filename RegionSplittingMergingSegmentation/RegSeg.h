#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace std;

struct Region
{
	mutable int id;
	int index;
	int width;
	int height;
	mutable int n;
	mutable float mean;

	Region(int id, int index, int width, int height, int n, float mean) :
		id(id),
		index(index),
		width(width),
		height(height),
		mean(mean) {}

	Region(int index, int width, int height, int n, float mean) :
		Region(-1, index, width, height, n, mean) {}

	Region() :
		Region(0, 0, 0, 0, 0) {}
};

struct Regions
{
	Region* data;
	int size;
};

class seq_seg
{
public:

	void run(uchar* data, int width, int height, float t_split, float t_merge);
	void displaySeg(uchar* d);
	void getRegions(Regions &rs) const;
	int getRegionsNumber();

protected:

	float threshold_split;
	float threshold_merge;
	uchar* data;
	int image_w;
	int image_h;
	Regions regions;

	void displayRegion(uchar* d, int reg_index);
	int getX(int index);
	int getY(int index);
	bool neighbors(Region &r1, Region &r2);
	void split(int start, int rWidth, int rHeight, vector<Region> &rs);
	float computeMean(int start, int rWidth, int rHeight);
	int computeSum(int start, int rWidth, int rHeight);
	float computeSD(float mean, int start, int rWidth, int rHeight);
	char* createRAG();
	void updateRAG(char* RAG, int x, int y);
	void merge(int curr, char* RAG);
	void merge(char* RAG);
	void mergeRegions(char* RAG, Region &r1, Region &r2);
	bool shouldMerge(Region& curr, Region& candidate, char* RAG);
	void initID();
};

class cuda_seg
{
public:

	void run(uchar* data, int width, int height, float t_split, float t_merge, bool reduction);
	void displaySeg(uchar* d);
	void getRegions(Regions &rs) const;
	int getRegionsNumber();

protected:
	float threshold_split;
	float threshold_merge;
	uchar* data;
	int image_w;
	int image_h;
	Regions regions;
	Region* d_rs;
	char* RAG;
	char* d_RAG;

	void displayRegion(uchar* d, int reg_index);
	void split(int start, int rWidth, int rHeight, vector<Region> &rs);
	float computeMean(int start, int rWidth, int rHeight);
	int computeSum(int start, int rWidth, int rHeight);
	float computeSD(float mean, int start, int rWidth, int rHeight);
	void createRAG();
	void updateRAG(char* RAG, int x, int y);
	void reduce();
	void merge(int curr, char* RAG);
	void merge(char* RAG);
	void mergeRegions(char* RAG, Region &r1, Region &r2);
	bool shouldMerge(Region& curr, Region& candidate, char* RAG);
	void initID();
};