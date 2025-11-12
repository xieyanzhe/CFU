#  Beyond Graph Priors: A Co-evolving Framework under Uncertainty for Enterprise Resilience Assessment
## Requirements
We implement CFU and other methods through the following dependencies:
- torch                             2.1.0+cu121
- numpy                          1.24.1
- dgl                                 2.4.0+cu121
- torch                               2.1.0+cu121
- torchaudio                     2.1.0+cu121
- torchdata                       0.11.0
- torchvision                    0.16.0+cu121
## Usage
Before running the code, ensure the package structure of PDU is as follows:
```text
CFU/
├── data/
│   ├── DBLP/
│   ├── mydata_2022/
│   ├── mydata_2023/
│   └── mydata_region/
│       └── data_readme
└── my-CFU/
    ├── pre_edge_parameter/
    ├── dataset_loader.py
    ├── our_model.py
    ├── pretrain_gcn.py
    └── train.py
```

### Train
```bash
python train.py
```
