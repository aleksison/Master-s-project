# Master-s-project
Multimodal merging: application to the ventriloquist effect
- The main file to run the DNF model is main_eval_new.py which is an adaptation of the original file main_eval.py for parameter optimization. Either the DNF parameters or the scaling factors for the input modalities(a_val for audio, v_val for visual modality)
- alais_burr_new.py contains the principal functions called in main_eval_new.py and also declares some important variables. Here we also create an array of the scenarios obtained from the raw data file: dataVentriloquie.xlsx
- In the evaluation folder sansbruit.py includes the results obtained from running the model where noise amplitude is set to 0. Mean12 for example stands for the results of mean localization position for tuple[1][2] where 1 is the position from auditory grid of values and 2 is the positon from the grid of visual values set in main_eval_new.py (a_val, v_val). This way all possible combinations are explored. V10, v8, etc stands for the results obtained by varying the visual  standard deviation declared in alais_burr_new.py. dnfstd1 -- position 1 in the array of tuples for different combinations of excitation width and inhibition width of the DNF. exc0 -- position 0 in the array of tuples exploring different combinations of excitation amplitude and excitation width of the DNF. All the arrays for grid search of different parameters/ pairs of parameters are declared in main_eval_new.py.
- evaluation/plots.py creates a scatter plot for a certain array of results and gives a loss value, so that we can compare how variations of the parameters affect the results.
- evaluation/pareto.py allows to choose the best solution based on the pareto front principle, when among the solutions there is no solution which is better in all the criteria(std and mean value).

Notion website with more details: https://aleksison.notion.site/Post-defense-work-432f8c8e8c94478d8db8995a2471eb7c
My email: aleksandralogino@gmail.com 
