#include "cnnModel.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <omp.h>	//for openMP

using std::vector;

cnnModelC::cnnModelC(parserC* p):
	mIteration_{50},
	mTrainSize_{900},
	mRow_{p->row()},
	mCol_{p->col()},
	mPcaData_{p->data()},
	mNumOfLev1_{5},
	mNumOfLev2_{5},
	mWeightVec1_{new double[(mCol_+1) * mNumOfLev1_]},
	mWeightVec2_{new double[(mNumOfLev1_+1) * mNumOfLev2_]},
	mWeightVec3_{new double[(mNumOfLev2_+1) * 3]}
{
	// Initial weight
	for(int i=0;i<(mCol_+1)*mNumOfLev1_;++i)
		mWeightVec1_[i]=0.0;
	for(int i=0;i<(mNumOfLev1_+1)* mNumOfLev2_;++i)
		mWeightVec2_[i]=0.0;
	for(int i=0;i<(mNumOfLev2_+1)*3;++i)
		mWeightVec3_[i]=0.0;
}
cnnModelC::~cnnModelC()
{
	delete[] mWeightVec1_;
	delete[] mWeightVec2_;
	std::cout<<"cnnModelC destructor\n";
}


void cnnModelC::showMatrix(double* data,int row,int col)
{
	for(int i=0;i<row;i++){
		for(int j=0;j<col;++j)
			std::cout<<data[i*col+j]<<" ";
		std::cout<<"\n";
	}
	std::cout<<"\n";
}
/**************************************************
 * cnnModelC::showPcaData()
 **************************************************/
void cnnModelC::showPcaData()
{
	int index{0};
	for(auto elem:mPcaData_)
	{
		std::cout<<std::fixed<<std::setprecision(6)<<elem;
		if((index++)%mCol_ == mCol_-1)
			std::cout<<"\n";
		else
			std::cout<<" ";
	}
}
void cnnModelC::modelTraining()
{
	// Algo
	int part = 8;
	double  targetData[part][3];
	double  outputData[part][3];
	double 	delta_k[part][3];
	double para=0.1;	
	double inputData[part][mCol_+1];
	double inputA1[part][mNumOfLev1_];
	double inputZeta[part][mNumOfLev1_+1];
	double inputA2[part][mNumOfLev2_];
	double inputZeta2[part][mNumOfLev2_+1];
	double derivedArrayFromA1[part][mNumOfLev1_];
	double derivedArrayFromA2[part][mNumOfLev2_];
	double diff_level3[part][(mNumOfLev2_+1)*3];
	double diff_level2[part][(mNumOfLev1_+1)*mNumOfLev2_];
	double diff_level1[part][(mCol_+1)*mNumOfLev1_];
	double sigmaWdeltaK[part][(mNumOfLev2_+1)];
	double sigmaWdeltaJ[part][(mNumOfLev1_+1)];
	double delta_j[part][mNumOfLev2_];
	double delta_m[part][mNumOfLev1_];

	double partWeightVec1[part][(mCol_+1) * mNumOfLev1_];
	double partWeightVec2[part][(mNumOfLev1_+1) * mNumOfLev2_];
	double partWeightVec3[part][(mNumOfLev2_+1) * 3];		

	int roundNumAll[part];
	int nData[part];
	// d
	int iterData[part];
	int whichClass[part];
	int nStart[part];
	int nEnd[part];
	omp_set_num_threads(2);
	#pragma omp parallel for 
	for (int partial = 0; partial < part; partial++) 
	{	
		int iteration = mIteration_;

		initPartWeight(partWeightVec1[partial], partWeightVec2[partial], partWeightVec3[partial]);

		// START ITERATION
		for(roundNumAll[partial] = 0 ; roundNumAll[partial] < iteration ; ++roundNumAll[partial])
		{
			//std::cout<<omp_get_thread_num()<<std::endl;
			//  nStart[partial] = 0;
			//  nEnd[partial]   = mTrainSize_;			
			
			nStart[partial] = partial * 100;
			nEnd[partial]   = partial * 100 + 900;

			// nStart[partial] = 900;
			// nEnd[partial]   = 1800;
			//for(int iterData=0 ; iterData<mTrainSize_ ; ++iterData)
			for(nData[partial] = nStart[partial] ; nData[partial] < nEnd[partial] ; ++nData[partial] )
			{
				if( nData[partial] >= mTrainSize_) 
				{
					iterData[partial] = mTrainSize_ - nData[partial] % mTrainSize_ ;
				}
				else iterData[partial] = nData[partial];
				//std::cout<< iterData[partial]<<std::endl;
				for(whichClass[partial] = 0; whichClass[partial]<3; ++whichClass[partial])
				{
					targetData[partial][0]=0.0;
					targetData[partial][1]=0.0;
					targetData[partial][2]=0.0;

					targetData[partial][whichClass[partial]]=1.0;
					initInputData(whichClass[partial], iterData[partial], inputData[partial]);	// assign data to inputData
					//std::cout<<"//-----------------------------------------------------------------------//\n";
					//std::cout<<"inputData\n";
					//showMatrix(inputData,1,mCol_+1);
					matrixMultiply(inputData[partial],
							partWeightVec1[partial],
							inputA1[partial],
							1,(mCol_+1),mNumOfLev1_);
					//std::cout<<"inputA1\n";
					//showMatrix(inputA1,1,mNumOfLev1_);

					sigmoidFun(inputA1[partial],mNumOfLev1_);
					//std::cout<<"tempZ\n";
					//showMatrix(inputA1,1,mNumOfLev1_);

					matrixAssign(inputZeta[partial],inputA1[partial],1,mNumOfLev1_);

					inputZeta[partial][mNumOfLev1_]=1.0;

					matrixMultiply(inputZeta[partial],
							partWeightVec2[partial],
							inputA2[partial],
							1, mNumOfLev1_+1 , mNumOfLev2_);
					//std::cout<<"inputA2\n";
					//showMatrix(inputA2,1,mNumOfLev2_);
					sigmoidFun(inputA2[partial],mNumOfLev2_);
					//std::cout<<"tempZ2\n";
					//showMatrix(inputA2,1,mNumOfLev2_);

					matrixAssign(inputZeta2[partial],inputA2[partial],1,mNumOfLev2_);

					inputZeta2[partial][mNumOfLev2_]=1.0;

					//std::cout<<"inputZeta2\n";
					//showMatrix(inputZeta2,1,mNumOfLev2_+1);
					matrixMultiply(inputZeta2[partial],
							partWeightVec3[partial],
							outputData[partial],
							1, mNumOfLev2_+1 , 3);
					sigmaExp(outputData[partial],3);
					transferOutput(outputData[partial],3);
					//std::cout<<"outputData:\n";
					//showMatrix(outputData,1,3);
					matrixMinus(outputData[partial],targetData[partial],delta_k[partial],3);
					//std::cout<<"delta_k\n";
					//showMatrix(delta_k,1,3);
					// start back propagation
					matrixMultiply(inputZeta2[partial],
							delta_k[partial],
							diff_level3[partial],
							mNumOfLev2_+1,1,3);
					//std::cout<<"diffLevel3\n";
					//showMatrix(diff_level3,mNumOfLev2_+1,3);

					getDerived(derivedArrayFromA2[partial],inputA2[partial],mNumOfLev2_);
					//std::cout<<"derivedArray A2\n";
					//showMatrix(derivedArrayFromA2,1,mNumOfLev2_);
					matrixMultiply(partWeightVec3[partial],delta_k[partial],sigmaWdeltaK[partial],mNumOfLev2_+1,3,1);
					//std::cout<<"sigmaWK\n";
					//showMatrix(sigmaWdeltaK,mNumOfLev2_+1,1);
					dotMultiply(derivedArrayFromA2[partial],sigmaWdeltaK[partial],delta_j[partial],mNumOfLev2_);
					//std::cout<<"delta_j\n";
					//showMatrix(delta_j,mNumOfLev2_,1);
					matrixMultiply(inputZeta[partial],delta_j[partial],diff_level2[partial],mNumOfLev1_+1,1,mNumOfLev2_);
					//std::cout<<"diff_level2\n";
					//showMatrix(diff_level2,mNumOfLev1_+1,mNumOfLev2_);
					getDerived(derivedArrayFromA1[partial],inputA1[partial],mNumOfLev1_);
					//std::cout<<"derivedArray A1\n";
					//showMatrix(derivedArrayFromA1,1,mNumOfLev1_);
					matrixMultiply(partWeightVec2[partial],delta_j[partial],sigmaWdeltaJ[partial],mNumOfLev1_+1,mNumOfLev2_,1);
					//std::cout<<"sigmaWJ\n";
					//showMatrix(sigmaWdeltaJ,mNumOfLev1_+1,1);
					dotMultiply(derivedArrayFromA1[partial],sigmaWdeltaJ[partial],delta_m[partial],mNumOfLev1_);
					//std::cout<<"delta_m\n";
					//showMatrix(delta_m,mNumOfLev1_,1);
					matrixMultiply(inputData[partial],delta_m[partial],diff_level1[partial],mCol_+1,1,mNumOfLev1_);
					//std::cout<<"diff_level1\n";
					//showMatrix(diff_level1,mCol_+1,mNumOfLev1_);

					for(int i=0;i<(mNumOfLev2_+1)*3;++i)
						diff_level3[partial][i] = para * diff_level3[partial][i];
					for(int i=0;i<(mNumOfLev1_+1)*mNumOfLev2_;++i)
						diff_level2[partial][i] = para * diff_level2[partial][i];
					for(int i=0;i<(mCol_+1)*mNumOfLev1_;++i)
						diff_level1[partial][i] = para * diff_level1[partial][i];
					matrixMinus(partWeightVec1[partial],diff_level1[partial],partWeightVec1[partial],(mCol_+1)*mNumOfLev1_);
					matrixMinus(partWeightVec2[partial],diff_level2[partial],partWeightVec2[partial],mNumOfLev2_*(mNumOfLev1_+1));
					matrixMinus(partWeightVec3[partial],diff_level3[partial],partWeightVec3[partial],3*(mNumOfLev2_+1));

				}// end-of-3 classes
			}// end-of-trained data
		}// end-of-iteration
		//updateWeight(partWeightVec1, partWeightVec2, partWeightVec3);	
	
	}// end-of-part
	// Get final weight
	for (int i = 0; i < part; i++) {
		for (int j = 0; j < (mCol_+1) * mNumOfLev1_; j++) mWeightVec1_[j] += partWeightVec1[i][j];
	}
	for (int i = 0; i < part; i++){
		for (int j = 0; j < (mNumOfLev1_+1) * mNumOfLev2_; j++) mWeightVec2_[j] += partWeightVec2[i][j];
	}
	for (int i = 0; i < part; i++){
		for (int j = 0; j < (mNumOfLev2_+1) * 3;j++) mWeightVec3_[j] += partWeightVec3[i][j];
	}	

	getFinalWeight(static_cast<double>(part));

}
void cnnModelC::modelTesting()
{
	double* inputData = new double[mCol_+1];
	double* inputA1 = new double[mNumOfLev1_];
	double* inputZeta = new double[mNumOfLev1_+1];
	double* inputA2 = new double[mNumOfLev2_];
	double* inputZeta2 = new double[mNumOfLev2_+1];
	double  targetData[3];
	double  outputData[3];
	double  error{0.0};

	for(int roundNum=0;roundNum<mIteration_;++roundNum)
	{
		for(int iterData=mTrainSize_ ; iterData<1000 ; ++iterData)
		{
			for(int whichClass = 0; whichClass<3; ++whichClass)
			{
				for(int n=0 ; n<3 ; ++n)
					targetData[n]=0.0;
				targetData[whichClass]=1.0;
				initInputData(whichClass, iterData, inputData);
				//std::cout<<"//-----------------------------------------------------------------------//\n";
				//std::cout<<"inputData\n";
				//showMatrix(inputData,1,mCol_+1);
				matrixMultiply(inputData,
						mWeightVec1_,
						inputA1,
						1,(mCol_+1),mNumOfLev1_);
				//std::cout<<"inputA1\n";
				//showMatrix(inputA1,1,mNumOfLev1_);

				sigmoidFun(inputA1,mNumOfLev1_);
				//std::cout<<"tempZ\n";
				//showMatrix(inputA1,1,mNumOfLev1_);

				matrixAssign(inputZeta,inputA1,1,mNumOfLev1_);


				inputZeta[mNumOfLev1_]=1.0;
				//std::cout<<"inputZeta\n";
				//showMatrix(inputZeta,1,mNumOfLev1_+1);


				matrixMultiply(inputZeta,
						mWeightVec2_,
						inputA2,
						1, mNumOfLev1_+1 , mNumOfLev2_);
				//std::cout<<"inputA2\n";
				//showMatrix(inputA2,1,mNumOfLev2_);
				sigmoidFun(inputA2,mNumOfLev2_);
				//std::cout<<"tempZ2\n";
				//showMatrix(inputA2,1,mNumOfLev2_);


				matrixAssign(inputZeta2,inputA2,1,mNumOfLev2_);


				inputZeta2[mNumOfLev2_]=1.0;

				//std::cout<<"inputZeta2\n";
				//showMatrix(inputZeta2,1,mNumOfLev2_+1);
				matrixMultiply(inputZeta2,
						mWeightVec3_,
						outputData,
						1, mNumOfLev2_+1 , 3);
				sigmaExp(outputData,3);
				transferOutput(outputData,3);
				if(compareMatrix(outputData,targetData,3))
					error+=1.0;
				//else std::cout<<whichClass<<std::endl;
			}
		}
	}
	std::cout<<"error: "<<error<<"\n";
	std::cout<<"accuracy: "<<error/static_cast<double>(1000-mTrainSize_)
							/(static_cast<double>(mIteration_)*3.0)
			 <<std::endl;
	delete inputData;
	delete inputA1;
	delete inputZeta;
	delete inputA2;
	delete inputZeta2;
}

/*************************************************
 * cnnModelC::initInputData()
 **************************************************/
void cnnModelC::initInputData(int whichClass, int iterData, double* inputData)
{
	double* pcaData = mPcaData_.data();
	for(int colCount=0;colCount<mCol_;++colCount)
		inputData[colCount] = pcaData[(whichClass+iterData*3)*mCol_ + colCount];
	inputData[mCol_]=1.0;
}
/**************************************************
 * cnnModelC::matrixMultiply
 * lhm : left hand matrix
 * rhm : right hand matrix
 * ans : ans array of double
 * row : the row# of ans
 * col : the col# of ans
 * mid : the middle# of two matrix
 **************************************************/
void cnnModelC::matrixMultiply(double* lhm,double* rhm,double* ans,int row,int mid, int col)
{
	for(int r=0;r<row;++r){
		for(int c=0;c<col;++c){
			ans[r*col+c]=0.0;
			for(int m=0;m<mid;++m)
			{
				ans[r*col+c] += lhm[r*mid+m] * rhm[m*col + c];
			}
		}
	}
}

/**************************************************
 * void cnnModelC::sigmoidFun(double* zeta,int size)
 **************************************************/
void cnnModelC::sigmoidFun(double* zeta,int size)
{
	for(int n=0 ; n<size ;++n){
		double expo = std::exp(-(zeta[n]));
		zeta[n] = 1.0 / (1.0 + expo);
	}
}

/**************************************************
 * void cnnModelC::sigmaExp(double* output,int size)
 **************************************************/
void cnnModelC::sigmaExp(double* output,int size)
{
	for(int n=0;n<size;++n){
		output[n] = std::exp(output[n]);
	}
	double expTotal=0.0;
	for(int n=0;n<size;++n){
		expTotal+=output[n];
	}
	for(int n=0;n<size;++n){
		output[n]=output[n]/expTotal;
	}
}
/**************************************************
 * Sinhong for parallel
 **************************************************/
void cnnModelC::initPartWeight(double* v1, double* v2, double* v3)
{
	for (int i = 0; i < (mCol_+1) * mNumOfLev1_; i++) {
		v1[i] = 1.0;
	}
	for (int i = 0; i < (mNumOfLev1_+1) * mNumOfLev2_; i++) {
		v2[i] = 1.0;
	}
	for (int i = 0; i < (mNumOfLev2_+1) * 3; i++) {
		v3[i] = 1.0;
	}
}

void cnnModelC::updateWeight(double* v1, double* v2, double* v3)
{
	for (int i = 0; i < (mCol_+1) * mNumOfLev1_; i++) {
		mWeightVec1_[i] += v1[i];
	}
	for (int i = 0; i < (mNumOfLev1_+1) * mNumOfLev2_; i++) {
		mWeightVec2_[i] += v2[i];
	}
	for (int i = 0; i < (mNumOfLev2_+1) * 3; i++) {
		mWeightVec3_[i] += v3[i];
	}

}

void cnnModelC::getFinalWeight(double part)
{
	for (int i = 0; i < (mCol_+1) * mNumOfLev1_; i++) {
		mWeightVec1_[i] /= part;
	}
	for (int i = 0; i < (mNumOfLev1_+1) * mNumOfLev2_; i++) {
		mWeightVec2_[i] /= part;
	}
	for (int i = 0; i < (mNumOfLev2_+1) * 3; i++) {
		mWeightVec3_[i] /= part;
	}	
}
/**************************************************
 * void cnnModelC::transferOutput(double* output,int size)
 **************************************************/
void cnnModelC::transferOutput(double* output,int size)
{
	if(size<1)
		return;
	int index{0};
	double maxData = output[0];
	for(int i=1;i<size;++i){
		if(output[i]>maxData)
		{
			index=i;
			maxData=output[i];
		}
	}
	for(int i=0;i<size;++i){
		output[i]=0.0;
	}
	output[index]=1.0;
}
/**************************************************
 * void cnnModelC::matrixMinus
 **************************************************/
void cnnModelC::matrixMinus(double* lhm,double* rhm,double* ans,int size)
{
	for(int i=0;i<size;++i)
		ans[i]=lhm[i]-rhm[i];
}

void cnnModelC::matrixAssign(double* lhm,double* rhm,int row,int col)
{
	for(int i=0;i<row;++i){
		for(int j=0;j<col;++j){
			lhm[i*col+j]=rhm[i*col+j];
		}
	}
}
void cnnModelC::getDerived(double* output,double* input,int size)
{
	for(int i=0;i<size;++i)
	{
		output[i] = input[i] * ( 1.0 - input[i]);
	}
}
void cnnModelC::dotMultiply(double* rhm,double* lhm,double* ans,int size)
{
	for(int i=0;i<size;++i)
		ans[i]=rhm[i]*lhm[i];
}
bool cnnModelC::compareMatrix(double* lhm ,double* rhm,int size)
{
	bool same{true};
	for(int i=0;i<size;++i)
		same &= (lhm[i]==rhm[i]);
	return same;
}
