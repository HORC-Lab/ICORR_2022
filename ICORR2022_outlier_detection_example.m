% Updated by Brad Hobbs, bwh@udel.edu, May 2022
% 
% This code chooses an example input, and runs the outlier detection algorithm
%
% Input: A - matrix of values with gait cycle periods in each row and with 
%            each gait cycle sample observation in each column
%
%        A - cell array of arbitrary size, with each cell consisting of a
%            matrix of values with gait cycle periods in each row and with
%            each gait cycle sample observation in each column

clear all; clear classes; close all; clc;

%% Choose input data
% Use given artificial data
A = importdata('artificial_data.mat');

% Use given example RVMO data
% A = importdata('subject6_RVMO_rigid.mat');

% Use custom data

%% Split data into non-outliers and outliers
[A_new,A_outs] = outlier_detection(A);

