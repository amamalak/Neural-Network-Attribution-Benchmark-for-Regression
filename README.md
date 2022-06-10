# Neural-Network-Attribution-Benchmark-for-Regression

"GEN_SYNTHETIC_BENCHMARK.m" is the main script to generate a synthetic benchmark dataset as descirbed in Mamalakis et al (2022).
"trainNN_XAI.ipynb" is the main script for building, training and explaining a fully-connected network to predict the synthetic y given the synthetic X.
"Plotting_Results.m" is the script that generates certain plots that appear in the paper. 

The scripts "PWL_gen.m", "figureHR.m" and "bluewhitered.m" are secondary functions that are called in the main scripts described above.  

The above scripts are user-friendly and with many instructions and can be used to generate totally new synthetic datasets as well as train new networks from scratch. For those who are interested in using exactly the same dataset as the one in Mamalakis et al., it can be found at: https://mlhub.earth/data/csu_synthetic_attribution

The trained network that was used by Mamalakis et al. is provided in the file "my_model.h5". 


## Citation
Mamalakis, A., I. Ebert-Uphoff, E.A. Barnes (2022) Neural network attribution methods for problems in geoscience: A novel synthetic benchmark dataset, Environmental Data Science, 1, E8, doi:10.1017/eds.2022.7. 
