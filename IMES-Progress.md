# Prgress Report, Oct 26, 2021
## Recent Progress
### paper: SMART SORTER - Deep RL
@Ardeshir, @Todd and @Masoud are working on both Unreal Engine4 and Unity3D to simulate the environment where the sorter wheels are controlable:
- **UE4**: We have now the Celluveyor hexagonal grids on UE4; @Todd is trying to write C++ scripts to control their wheels. [we left UE4]
- **Unity**: We wrote some scripts to control the GameObjects on Unity; We're trying to find a way to import CAD files to Unity.[a script to create random objects(random boxes) is written. A script to transform the objects in a given path is written.]

progress: **20%**

### paper: Transfer Learning for Condition Monitoring of Rotating Machinery
@Masoud imported and made the dataset ready to implement. 
@Todd is replicating WC-RNN and @Ardeshir is replicating an imporved CNN to train on the dataset.
@Seyi is working on the feature engineering / signal processing part: EEMD technique is getting ready to implement.

progress: **15%**

### paper: Generative DL for Condition Monitoring of Rotating Machinery
@Masoud imported and made the dataset ready to implement.
@Maryam has developed a preprocessing tool for normalizing the signals with different RPMs.
@Maryam imports her VAE model to the diagnosis architecture panel.

progress: **15%**

## Next Stage
### paper: SMART SORTER - Deep RL
@Ardeshir and @Todd find a way to import CAD files to Unity3D.

### paper: Transfer Learning for Condition Monitoring of Rotating Machinery
@Seyi implements FFT and other transform techniques to extract useful info out of the signals.
@Masoud, @Ardeshir and @Todd replicate some state-of-the-art Transfer Learning architectures.

### paper: Generative DL for Condition Monitoring of Rotating Machinery
@Masoud and @Maryam are developing a GAN architecture for it to test against the VAE.

## Open Issues
1- CAD files are not going to be imported directly to Unity. Solutions:
- Converting the .DWG to .OBJ
- Purchasing the CAD importer adds-on.

