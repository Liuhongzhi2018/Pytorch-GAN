
close all;

I = im2double(imread('epoch186_real_A.png'));

tic;
res = RollingGuidanceFilter(I,3,0.05,4);
toc;

figure,imshow(I);
figure,imshow(res);
figure,imwrite(res,'output.png')