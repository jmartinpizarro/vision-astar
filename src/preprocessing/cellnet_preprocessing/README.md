# Why does the CellNet another different type of preprocessing? #

This happens because of the approach decided to use for this specific type of neural network. As this networ is not processing the entire image. but the contents of each cell, data (characters inside each cell) must be preprocessed individually.

This means that:
1. This network is similar to, for example, a generalistic network that is able to process the MINST dataset (inside numbers we are working with letters).
2. Size cannot be determined, but to be given as an input in order to be able to generate the output correctly.

# Why the dataset is firstly preprocessed (normalised) and then splitted into train/test #

As we are working with deterministic transformations, there is no problem. If we included in the preprocessing some rotation of images (which we are not, we are doing that in the generation of the dataset), things would be different.