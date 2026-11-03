import os

MARK = '[RUN GUIDE]'
BAR = '=' * 72

HEADERS = {
'train_metrics.py': """
train_metrics.py  -  train ONE model variant, log metrics every epoch

Run in a WSL terminal, from ~/BERT-TFBS (project is /workspace in the container).

  0. start the container (once per boot):
       docker compose up -d
  1. train in the background, output to a log file:
       docker compose exec -d bert-tfbs sh -c "python -u train_metrics.py --arch full --tag full --seed 41 > /workspace/results_full_41.log 2>&1"
     --arch   full | v1 | v2 | resnet   which model to train
     --tag    label for output file names (keep it equal to --arch)
     --seed   random seed (use 41, 42, 43)
     --epochs default 15
  2. watch progress (Ctrl+C stops watching, NOT training):
       tail -f results_full_41.log
     is it still running? (no output = finished):
       docker compose exec bert-tfbs pgrep -af train_metrics
  3. outputs:
       Dataset/<folder>/model_<tag>_seed<N>.pth    weights (overwritten every epoch)
       results/metrics_<tag>_seed<N>.csv           per-epoch train/test metrics
  4. then test it:  python test.py --arch <arch> --weights model_<tag>_seed<N>.pth

Train ONE job at a time on a 6 GB GPU.
""",
'train.py': """
train.py  -  ORIGINAL author's training script (reference / baseline only)

Prefer train_metrics.py (adds metrics, eval mode, seeds, CSV logs).
If you do run this one, from ~/BERT-TFBS in a WSL terminal:
     docker compose up -d
     docker compose exec -d bert-tfbs sh -c "python -u train.py > /workspace/train.log 2>&1"
     tail -f train.log
It saves Dataset/<folder>/model.pth  (this overwrites a copy of the shipped weights).
""",
'script/metrics.py': """
script/metrics.py  -  metric helpers (library file: do NOT run it directly)

Imported by train_metrics.py and test.py.
  MetricTracker : collects predictions, .compute() returns acc, precision, recall,
                  specificity, F1, MCC, ROC-AUC, PR-AUC, loss, tn/fp/fn/tp
  append_csv    : append one row to a CSV (writes the header if the file is new)
""",
'script/model_resnet.py': """
script/model_resnet.py  -  BERT-TFBS with the CNN front end replaced by a ResNet
(library file: do NOT run it directly; chosen with --arch resnet)

Only the CNN front end (conv1, parallel branches, conv3) is replaced. CBAM and the
output module are inherited unchanged from script/model.py.

  Quick shape check (from ~/BERT-TFBS, container up):
     docker compose exec bert-tfbs python -c "import torch; from script.model_resnet import ResNetCNNNET; print(ResNetCNNNET(30).eval()(torch.randn(4,30,768)).shape)"
     expected: torch.Size([4, 2])

  Train / test:
     docker compose exec -d bert-tfbs sh -c "python -u train_metrics.py --arch resnet --tag resnet --seed 41 > /workspace/results_resnet_41.log 2>&1"
     docker compose exec bert-tfbs python test.py --arch resnet --weights model_resnet_seed41.pth --out results/test_resnet_seed41.csv

  Depth: n_blocks in Bert_Blend_CNN.__init__ (default 2). Use a new --tag when you change it.
""",
'docker-compose.yml': """
docker-compose.yml  -  GPU container for BERT-TFBS (project mounted at /workspace)

Run in a WSL terminal, from ~/BERT-TFBS:
     docker compose up -d --build      build the image (first time / after Dockerfile or requirements change) and start
     docker compose up -d              start (after a reboot)
     docker compose ps                 is it running?
     docker compose exec bert-tfbs bash   open a shell inside the container
     docker compose down               stop and remove the container
Check the GPU:  docker compose exec bert-tfbs python -c "import torch; print(torch.cuda.is_available())"
""",
'Dockerfile': """
Dockerfile  -  Python 3.9 + torch 1.12.0 (CUDA 11.3) + transformers 4.22.2
Build/start with docker-compose.yml:   docker compose up -d --build
Rebuild only after changing this file or requirements.txt.
""",
}

for path, text in HEADERS.items():
    if not os.path.isfile(path):
        print('skip (missing):', path)
        continue
    raw = open(path, 'rb').read()
    if MARK.encode() in raw[:300]:
        print('skip (already has header):', path)
        continue
    nl = '\r\n' if b'\r\n' in raw else '\n'
    lines = ['# ' + BAR, '# ' + MARK] + [('# ' + l).rstrip() for l in text.strip('\n').split('\n')] + ['# ' + BAR, '']
    open(path, 'wb').write((nl.join(lines) + nl).encode() + raw)
    print('header added:', path)
