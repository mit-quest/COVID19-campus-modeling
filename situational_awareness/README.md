# Situational Awareness

## Instructions to run situational awareness

To run this simulation, do it from the top level directory e.g.

`➜ python situational_awareness/situational_analysis.py --config-file situational_awareness/simulation_configuration.json`

On a 96-core GCP VM, running 50 replications in parallel (as already done in the code), this takes ~2min.


As detailed in the technical report, the situational awareness system is meant to be run hourly and provides decision makers with hourly predictions of building usage statistics and alerts them of any buildings where occupancy has reached a high thresholds of density.

Structure of this python file:
- reading simulation configuration json is done by the class `__init__`. Then everything is run through the `run function`, namely:
  - reading data: card reader data, WiFi data and class schedule data present in `situational_awareness/data/` is read:
    - Card reader data is processed in `self.manage_card_reader`, and from the data various distributions are created.
    - class schedule to be injected is read in `self.read_schedule_injection`
    - assignment distritubitions are created in `self.make_assignment_buildings_count`
    - transition distributions are created in `self.make_buildings_transition_pmf`
    - duration distribution is created in `self.make_durations_pmf_dict`
  - simulations are carried out in `self.make_output_past_future_dict` where:
    - a number <self.N_replication> of parallel (using multiprocessing) replications are run in which inflow, outflow and occupancy are sampled
    - It is important to note that the last time for which we have card reader data (based on when this script was run, via `metadata_dict['historical_parameters']['interval_date_end']`) splits the sampling into a past and future window (whose lengths are defined via `simulate_n_hours_window_backward` and `simulate_n_hours_window_forward` which are currently defined to 2 weeks x 7 days x 24 hours.
    - inflows (from arrival times) and occupancy are sampled from empirical distributions built from the data above. Sampling is done as such:
      - For the past data, inflow is simply read from the data and not sampled. For future data,
      inflow is sampled based on hour of the week (7 days x 24 hours) from the past 4 weeks of data (which are converted to a simple gaussian). If `inject_in_person_class_headcount` is `True`, then spring class schedule headcounts are injected with a ramp in the arrival/inflow model.
      - Occupancy and outflow are sampled in both past and future. If `building_transition_function` is
      `run_transition_swipe_to_assigned_building`, then people's transitions are simulated from the building they swipe in (
      where arrival is recorded) to the building they end up with (destination ie assignment building).
  - Based on these simulations, `self.make_output_json` creates human and UI (as run in `frontend/`) readable jsons:
    - Statistics such as mean, standard deviation and 10/50/90 are calculated and output in a json format.
    - The data can also be plotted by setting `PLOT` to True through a number of different plot functions.
  - To run predictive checks, `self.create_ground_truth_comparison` allows us to compare the statistics from different replications (
  e.g. the distribution of inflows) for a certain time range (as defined in `self.analysis_parameters[
  'debug_parameters']['ground_truth_day']` to be compared to the manually measured statistics.


### Execution flag checklist
These flags in the `config.json` file modify the execution of how situational awareness is ran:
- `inject_in_person_class_headcount`: this code is used to inject inflow, occupancy and outflow from class schedules.
- `plot`: for plotting.
- `limited_buiding_list`: further simplification for faster debugging. This makes it so that occupancy simulation is only done on a limited set of buildings (as opposed to all 206 of them) defined in `debug_building_list`.
- `ground_truthing`: this is used to plot histograms of predicted distributions of occupancy and inflow vs. some externaly valid numbers (in our cases, people standing outside of buildings and recording inflow and occupancy)


## Alert creation
`create_alerts.py` is used to create alerts based on whether building usage statistics have reached a critical threshold (therefore leading to higher risk). `OCCUPANCY_ALERT_THRESHOLD` is set to be the threshold at which at alert is created if `estimatedOccupancy` > (`threshold` * `totalAssigned`).  `totalAssigned` is read from `situational_awareness/data/synthetic_building_assignment_totals.json`.


It can be run from the top-level directory as such:
`➜ python situational_awareness/create_alerts.py`

## How to contribute
If you'd like to contribute improvements to this code, please make sure that the changes do not break on the sythetic dataset.

