/*
 * NumberGenerator.h
 *
 *  Created on: Apr 14, 2019
 *      Author: daniele
 */
#include <vector>

using namespace std;


#ifndef NUMBERGENERATOR_H_
#define NUMBERGENERATOR_H_

class NumberGenerator {
public:
	//methods callable from the external
	NumberGenerator(int lowerBound, int upperBound);
	void initializeNumbers(float milliseconds);
	virtual ~NumberGenerator();
	void computeMeanVarAllAxes();

private:
	float generateRandomFloatInRange();
	float computeMeanVector(vector<float> inputVec);
	float computeSTDVector(vector<float> inputVec, float mean);
	float computeRMS(vector<float> inputVec);


	int lowerBound;
	int upperBound;

	float meanX;
	float meanY;
	float meanZ;

	float stdX;
	float stdY;
	float stdZ;

	float rmsX;
	float rmsY;
	float rmsZ;

	vector<float> bufferX;
	vector<float> bufferY;
	vector<float> bufferZ;
};

#endif /* NUMBERGENERATOR_H_ */
