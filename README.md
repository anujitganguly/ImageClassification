# ImageClassification
Beginner project in ML to classify image using CNN and tensorflow(GPU)

# PLEASE NOTE: This project worked perfectly only once after trying multiple times, but as it worked fine in older GPUs supporting earlier version of CUDA (say 9.0) therefore I am sharing this. The problem is mainly because of CUDNN as tensorflowGPU is used in the project. I tried almost every possible way mentioned in the internet but didn't found a perfect description about why it worked only once and then again the problem "CUDNN failed to initialize" was encountererd. IF SOMEONE WITH THE BELOW MENTIONED SYSTEM CAN FIND A SOLUTION THEN PLEASE HELP.

System Details: Windows 10-64bits, Intel-9thGen-i5, NvidiaGeforce-GTX1650-4gb, Tensorflow2.0(GPU), CUDA-v10.0, cudnn-v7.4.1(for CUDA-v10.0), RAM 8gb, Spyder(Anaconda3)

Project Description: This is a modified work which I learnt from the given link https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd This project is an entry level guide for those who are working with Tensorflow GPU version and facing problems as some basic codes are changed to run a code in the CPU version. It works perfectly fine with GPUs that support CUDA-v9.0 and Tensorflow-v1.8. The dataset that is used in this project is also taken from the above link. The output that I recieved only once predicted perfectly the result. I am still working on it to find the exact reasons for it's incompatability with latest systems, however as per NVDIA forums many people had faced the same problem.

Conclusion: Can be used for learning purpose by beginners.
