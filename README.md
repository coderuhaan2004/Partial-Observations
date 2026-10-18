## Run Primary and Secondary User
- Set the configurations of the experiment in `experiment-config.yaml`
- Run the python scripts on different devices accordingly.
```bash
python src/primary-user.py <pluto-uri>
python src/secondary-user.py <pluto-uri>
```

Primary-user occupation events are also saved continuously on the primary
user's machine as JSON Lines at `data/primary/<pu-id>.jsonl`. Each event is
flushed and synced before transmission starts, so turning off the fusion
server or stopping the process does not discard already recorded events.
The path can be changed with `PrimaryDataset.storage_path` in
`src/config.yaml`.

Secondary-user IQ frames are saved continuously as JSON Lines at
`data/secondary/<su-id>.jsonl` and are also sent to the fusion server by the
existing background client. Local writes are flushed and synced before the
receiver continues. Change the path with `SecondaryDataset.storage_path`.
## What more (Priority Ordered) ?
- [ ] Implement the exact same hardware experiment on MATLAB simulation with path loss models taken from the [Paper](https://ieeexplore.ieee.org/document/10505885)
- [ ] Add learnable graph structure by replacing the fixed distance matrix to learnable distance matrix.
- [x] Need to design how SU gets to know which channel the assigned PU is occupying.
- [x] Information from all the SUs will be given to the fusion server. So fusion server needs to be implemented.
- [ ] 64 frequency points for each band. (Probably achieved by OFDM signal)
- [x] Asynchronous file I/O if computation within while loop of SU is taking significant time.