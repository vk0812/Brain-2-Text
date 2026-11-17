# Concepts

Groups all the notions that required additional research to understand.

## Table of Contents

- [Electroencephalography (EEG)](###electroencephalography-(eeg))

- [Haemodynamic Response Function (HRF)](###haemodynamic-response-function-(hrf))

- [Stimuli](###stimuli)

- [Functional Magnetic Reasonance Imaging (fMRI)](###functional-magnetic-resonance-imaging)

### Electroencephalography (EEG)

Listen to electrical activity under someone's scalp using sensors.

A neuron firing alone doesn't produce enough electricity to be detected.

Millions of neurons firing at the same time = group of people clapping in a stadium.

**EEG signals:**

![eeg_signals_table](/Cheatsheets/Assets/eeg_signals_table.png)


### Haemodynamic response function (HRF)

Model of how the **BOLD** signal evolves over time following neural activity.

### Stimuli
Detectable change in the environment $\to$ organism biological or physiological response. 

This can include sensory inputs that trigger reactions in the nervous system like:

- light

- sound

- touch

### Functional Magnetic Resonance Imaging

**Chain of events:**

1. Brain region becomes active

2. Neurons fire more intensely

3. Local blood vessels dilate and increase blood flow in the region

4. Fresh supply of **oxygenated** hemoglobin is delivered

`Seiji Ogawa, 1990`:

- **Oxygenated hemoglobin** = diamagnetic (weakly _repelled by_ magnets)

- **Deoxygenated hemoglobin** = paramagnetic (weakly _attracted to_ magnets)

#### fMRI machine

Giant magnet operating at:

$$
[1,5\to7] Tesla \ge [30 000\to140000] * \text{Earth's magnetic field}
$$

Scanner $\to$ Magnetic field $\to$ Ratio of **oxygenated** vs **deoxygenated** hemoglobin.

It does so **voxel** by voxel.

**Voxel:** 3D equivalent of a pixel. $1 \to 3$ mm on each side.

This measurement is called **BOLD signal** (same as dataset?).

fMRI detects the response to brain activity.

It can take $5\to 6$ seconds after the event that triggered it.

Slow but super precise, up to a mm on the whole brain.

**Note:**

fMRI = big machine that costs a ton.

EEG = wearable device.

### GLM beta estimates

Relative weights of different predictors in modeling the brain's response to stimuli.

Help quantify how strongly each modeled effect contributes to the measured brain activity over time.

How much of the stimuli can we attribute to this image.

The answer requires fitting a statistical model to the time series.