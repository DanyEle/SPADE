//============================================================================
// Name        : SPADE_Analysis.cpp
// Author      : Daniele Gadler
// Version     : 0.1
// Copyright   : 
//============================================================================

#include <iostream>
#include <vector>
#include "NumberGenerator.h"

using namespace std;

int main() {

	NumberGenerator numGenerator(-10 , 10);

	//now use the number generator to put dummy data into the vector
	numGenerator.initializeNumbers(500);

	numGenerator.computeMeanVarAllAxes();

	return 0;
}
