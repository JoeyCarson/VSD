/*
 * ViolenceModel.cpp
 *
 *  Created on: Mar 16, 2016
 *      Author: josephcarson
 */

#include "ViolenceModel.h"
#include <ejdb/ejdb.h>
#include "opencv2/opencv.hpp"

#define VIOLENCE_MODEL_DB_NAME "violence_model.db"

ViolenceModel::ViolenceModel()
: ejdb(NULL)
{
	ejdbInit();
}

void ViolenceModel::ejdbInit()
{
	EJDB * tempEJDBPtr = NULL;

	if ( !ejdb ) {

		tempEJDBPtr = ejdbnew();

		if ( !tempEJDBPtr ) {
			std::cerr << "Failed Instantiating EJDB object.";
		} else if ( !ejdbopen(tempEJDBPtr, VIOLENCE_MODEL_DB_NAME, JBOWRITER | JBOCREAT) ) {
			std::cerr << "Failed opening EJDB database.";
			free(tempEJDBPtr); tempEJDBPtr = NULL;
		} else {
			// Success.
			std::cout << "Database opened.";
			this->ejdb = tempEJDBPtr;
		}
	}
}



ViolenceModel::~ViolenceModel() {
	// TODO Auto-generated destructor stub
}

