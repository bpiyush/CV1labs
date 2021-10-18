## CV1 Lab Project: Part 1

### Demo

In order to run through a demo, please follow the steps:

1. Setup the `conda` environment with required (standard) packages
2. Run the classifier
   ```bash
   cd /path/to/labfinal1/
   python classify.py -n 500 -d sift
   ```
   Here, `-n 500` denotes the number of clusters to use, `-d sift` denotes using SIFT for extracting descriptors.

### Results

For $K \in \{500, 1000, 2000\}$ and with default SVM hyper-parameters, we obtain the following average precision values on the test set. Note that `mean` here means the usual `mAP`.

|                   |   airplane |     bird |     car |    horse |     ship |     mean |
|:------------------|-----------:|---------:|--------:|---------:|---------:|---------:|
| $K = 500$ |   0.623879 | 0.473548 | 0.61977 | 0.606844 | 0.599587 | 0.584726 |
| $K = 1000$ |   0.625489 | 0.473098 | 0.59616 | 0.609306 | 0.577168 | 0.576244 |
| $K = 1000$ |   0.607163 | 0.454526 | 0.589225 | 0.594055 | 0.552394 | 0.559472 |

The observed classification accuracy is as follows:
| $K$ | Accuracy | mAP |
|:----|---------:|----:|
| $500$ | 0.5485 | **0.584726** |
| $1000$ | **0.5545** | 0.576244 |
| $2000$ | 0.5335 | 0.559472 |

See the report for more results and analyses.

### Further experiments and analysis

We perform additional analyses in notebook in the `notebooks/` folder.
