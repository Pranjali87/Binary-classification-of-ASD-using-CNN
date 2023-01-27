# Binary-classification-of-ASD-using-CNN
I downloaded KKI dataset from ABIDE. I have downloaded raw MRI images and preprocessed it in python using nilearn packages.
This project classifies ASD vs TD from sMRI and rsMRI on KKI image dataset(ABIDE). 
Total 55 subject,22 ASD and 33 Typical control.
I use VGGNET16 architecturefor classification. preprocess.py contain preprocessing of anat and rest images.
In preprocessing part , I applied regular steps of preprocessing along with ICA and dict learning. here i took 5 components.
module1.py preprocess all the anat and rest images.
module2.py separate images of autism in 1 folder and TD in second folder with the help of phenotypic data i.e csv file. 
Module 3.py use cnn and classifies ASD vs TD with 63.75 % accuracy at 10 epoch.
