

#include "RollingGuidanceFilter.h"

int main(){

	String name = "./imgs/image.png";

	Mat img = imread(name);

	if(img.empty()){
		printf("No such file.\n");
		getchar();
		exit(1);
	}

	clock_t startTime = clock();
	Mat res = RollingGuidanceFilter::filter(img,3,25.5,4);
	printf("Elapsed Time: %d ms\n",clock()-startTime);

	imshow("img",img);
	imshow("res",res);
	waitKey();

	return 0;
}