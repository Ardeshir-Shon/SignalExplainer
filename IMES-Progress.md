# Progress Report, November 25, 2021
## Recent Progress
### paper: SMART SORTER - Deep RL
- @Ardeshir and @Todd formulated the problem, mathematically and created power point presentation with defining project details.
- @Ardeshir started introduction for paper.
- @Todd started writing the problem formulation for paper.

progress: **35%**

### paper: Generative DL for Condition Monitoring of Rotating Machinery
- @Maryam wrote a draft of the introduction
- @Masoud checked the dual-GAN performance on the dataset. -- **issue**: Dual-GAN had a very poor performance.
- @Ardeshir and @Masoud are developing a Wasserstein-GAN with GradientPenalty (WGANGP).
- @Seyi is providing the feature extraction techniques.
- @Seyi is writing the feature engineering part of the paper.

progress: **40%**

# Progress Report, Oct 26, 2021
## Recent Progress
### paper: SMART SORTER - Deep RL
@Ardeshir, @Todd and @Masoud are working on both Unreal Engine4 and Unity3D to simulate the environment where the sorter wheels are controlable:
- **UE4**: We have now the Celluveyor hexagonal grids on UE4; @Todd is trying to write C++ scripts to control their wheels. [we left UE4]
- **Unity**: We wrote some scripts to control the GameObjects on Unity; We're trying to find a way to import CAD files to Unity.[a script to create random objects(random boxes) is written. A script to transform the objects in a given path is written.]
- @Ardeshir and @Todd have fully defined the initial problem, including the state estimation variables, action space, and reward function.
- @Ardeshir and @Todd prepared a preliminary presentatioon describing the project in detail.

progress: **25%**

### Comments
- **C**: @Masoud I'm a little concerned about the deadline for submission. I noticed you've made 25% progress so far, and I feel we should accelerate a little to meet the 7 and 30 December deadlines according to the publication plan. What do you think?
- I have also checked Overleaf, do you think the team is ready to start writing the introduction, overview, backgorund, and some other sections of the paper?
- **R**:

### paper: Transfer Learning for Condition Monitoring of Rotating Machinery
@Masoud imported and made the dataset ready to implement. 
@Todd is replicating WC-RNN.
@Masoud developed an imporved CNN, AE and a ConvLSTM to train on the dataset.
@Ardeshir is developing a Transfer Learning model.
@Seyi is working on the feature engineering / signal processing part: EEMD technique is ready; FFT is also under investigation.

progress: **25%**

### paper: Generative DL for Condition Monitoring of Rotating Machinery
@Masoud imported and made the dataset ready to implement.
@Maryam has developed a preprocessing tool for normalizing the signals with different RPMs.
@Maryam imports her VAE model to the diagnosis architecture panel.

progress: **15%**

## Next Stage
### paper: SMART SORTER - Deep RL
@Ardeshir and @Todd are writing the formulation of the problem.

### paper: Transfer Learning for Condition Monitoring of Rotating Machinery
@Seyi implements FFT and other transform techniques to extract useful info out of the signals.
@Masoud, @Ardeshir and @Todd replicate some state-of-the-art Transfer Learning architectures.

### paper: Generative DL for Condition Monitoring of Rotating Machinery
@Masoud and @Maryam are developing a GAN architecture for it to test against the VAE.

## Open Issues
1- CAD files are not going to be imported directly to Unity. Solutions:
- Converting the .DWG to .OBJ [solved]
- Purchasing the CAD importer adds-on. [a month-trial can do the job ==> solved]

