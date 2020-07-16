# EMG-CNN
1. Make sure to download Stockwell Transform for MATLAB (st.m)
2. Import your time-series EMG signal to the MATLAB environment with each fatigue level its own vector/matrix (NOT table)
3. Run dsp.m, it will output folders of each fatigue level containing individual windows
4. Ensure 'high_dir' and 'low_dir' in model.py point to these folders
5. Run model.py to split data, train the model, and evaluate the model


### Notes
- I've gotten the model to be pretty accurate even with a ridiculously small window (currently 40 samples, equivalent to 1/100th of a second of data)
- After exponential drops around 10 epochs the model seems to linearly improve until asymptoting (and likely overfitting) around 200 epochs
