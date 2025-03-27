# TIFocus

A tool to analyze TIF stacks in order to extract the sharpest frame and perform quality control for high-throughput imaging.
The tool calculates the laplacian variance for each image in the stack and outputs the slice of the sharpest image.
Experiments have shown that Laplacian-based operators have the best overall performance at normal imaging conditions.
Reference: https://doi.org/10.1016/j.patcog.2012.11.011

## Planned features
- Plotting the slice number of the sharpest slice for a batch of stacks. This allows for comparison of different focus strategies
- Support for Multichannel-TIF files
- Addiional statistics
- Additional quality control metrics
- Additional batch processing features
