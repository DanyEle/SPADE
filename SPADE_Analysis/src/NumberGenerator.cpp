/*
 * NumberGenerator.cpp
 *
 *  Created on: Apr 14, 2019
 *      Author: daniele
 */
#include <iostream>
#include <vector>
#include <chrono>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <cmath>


#include "NumberGenerator.h"

using namespace std;

NumberGenerator::NumberGenerator(int lowerBoundIn, int upperBoundIn) {
	cout << "Initializing OBJ" << endl;

	this->lowerBound = lowerBoundIn;
	this->upperBound = upperBoundIn;

	this->meanX = 0;
	this->meanY = 0;
	this->meanZ = 0;

	this->stdX = 0;
	this->stdY = 0;
	this->stdZ = 0;

	this->rmsX = 0;
	this->rmsY = 0;
	this->rmsZ = 0;
}

//This method fills in the vectors in the instance level with values acquired by
//a simulated accelerometer within 5 seconds
void NumberGenerator::initializeNumbers(float milliseconds)
{
	//firstly, seed the random number generator (just needs to be done once before generating numbers)
	srand (static_cast <unsigned> (time(0)));


	 auto start = chrono::system_clock::now();

	 bool continueGeneration = true;

	 int i = 0;
	 while(continueGeneration)
	 {
		//generate random numbers simulating accelerometer.
		float randomNumX = this->generateRandomFloatInRange();
		float randomNumY = this->generateRandomFloatInRange();
		float randomNumZ = this->generateRandomFloatInRange();

		//now check if enough time has elapsed
		auto end = chrono::system_clock::now();

	    std::chrono::duration<float, std::milli> duration = end-start;
	    using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;

	    auto f_ms = FpMilliseconds(end - start);

	    //time has elapsed
	    if(f_ms.count() >= milliseconds)
	    {
	    	continueGeneration = false;
	    	cout << "Need to stop generating " << endl;
	    }
	    //time has not yet elapsed
	    else
	    {
	    	this->bufferX.push_back(randomNumX);
	    	this->bufferY.push_back(randomNumY);
	    	this->bufferZ.push_back(randomNumZ);
	    	i++;
	    	std::cout << "Time in milliseconds, using floating point rep: " << f_ms.count() << '\n';
	    }
	 }
	 cout << "Generated " << i << " random numbers " << endl;
	//generate values in the range of time passed
}

float NumberGenerator::generateRandomFloatInRange()
{
	float random = ((float) rand()) / (float) RAND_MAX;
	float diff = this->upperBound - this->lowerBound;
	float r = random * diff;

	return r;
}


void NumberGenerator::computeMeanVarAllAxes()
{

	this->meanX = this->computeMeanVector(this->bufferX);
	this->meanY = this->computeMeanVector(this->bufferY);
	this->meanZ = this->computeMeanVector(this->bufferZ);

	this->stdX = computeSTDVector(this->bufferX, meanX);
	this->stdY = computeSTDVector(this->bufferX, meanX);
	this->stdZ = computeSTDVector(this->bufferX, meanX);

	this->rmsX = computeRMS(this->bufferX);
	this->rmsX = computeRMS(this->bufferY);
	this->rmsX = computeRMS(this->bufferZ);


}

float NumberGenerator::computeMeanVector(vector<float> inputVec)
{
	//cout << "MEAN" << inputVec.front() << endl;
	int amountElements = inputVec.size();
	float sumElements = std::accumulate(inputVec.begin(), inputVec.end(), 0);
	return sumElements / amountElements;
}

float NumberGenerator::computeSTDVector(vector<float> inputVec, float mean)
{
	//cout << "STD" << inputVec.front() << endl;
	//vectorial operation, need to subtract each element by the mean (in place)
	std::transform(inputVec.begin(), inputVec.end(), inputVec.begin(),
	          bind2nd(std::minus<float>(), mean));

	//now need to sum all the elements
	float sumElements = std::accumulate(inputVec.begin(), inputVec.end(), 0);

	//and need to divide by the amount of elements
	float divElements = sumElements / inputVec.size();

	//and finally perform the square root and return that
	return sqrtf(divElements);
}

float NumberGenerator::computeRMS(vector<float> inputVec)
{
	//cout << "RMS" << inputVec.front() << endl;
	//need to compute the square root of every single input element in the vector.
    std::transform(inputVec.begin(), inputVec.end(), inputVec.begin(), [](int x){return x*x;});

    //now need to sum all the elements
	float sumElements = std::accumulate(inputVec.begin(), inputVec.end(), 0);

	//and need to divide by the amount of elements
	float divElements = sumElements / inputVec.size();

	//and finally perform the square root and return that
	return sqrtf(divElements);

}


NumberGenerator::~NumberGenerator() {
	// TODO Auto-generated destructor stub
}

