# Model
- The model calculate the infection probability of every commute trips by different travel modes (see config.json "mode_list"): 'MBTA','Drive own vehicle','Bike','Walk','Taxi/Uber/Lyft','other'
- The infection model is based on Wells-Riley equation.
-- Train: length 21.3m width:2.4m height:2.18m (https://bc.mbta.com/business_center/bidding_solicitations/pdf/MBTA%20RO%20Technical%20Specification%20October%2022%202013.pdf). air change per hour: 8(https://787updates.newairplane.com/getmedia/88faa01a-d578-4e1a-be44-3a177c8ef50f/Air-change-rates-inplanes).
-- Bus: length 12m width:2.6m height: 2.1m (https://en.wikipedia.org/wiki/MBTA_bus) air change per hour: 8 (document/10.1.1.500.3711.pdf)
-- car: volume 5m^3; air change per hour: 54 (https://www.nature.com/articles/7500601.pdf?origin=ppub)

- pulmonary ventilation:
-- bicyle 23.5L/min
-- car 11.8L/min
-- bus, train: 12.7L/min
(Minute ventilation of cyclists, car and bus passengers: an experimental study)
-- walk 1.5m/s: 28L/min (https://www.nature.com/articles/s41598-017-05068-8/tables/1)

# Model structure and functions:
- For each individual and each of his/her commuting trip, we judge the travel mode first, and then based on the travel mode, we have five functions:
- self.infected_by_public_transit()
- self.infected_by_drive()
- self.infected_by_bike()
- self.infected_by_walk()
- self.infected_by_carpool()
- self.infected_by_other()

# Data
- PT_trips_records.csv, Bike_trips_records, Walk_trips_records. These three files provide routes for different modes given an Origin-Destination (OD) pair.
- routes_subway.csv routes_bus.csv. These two files provide the exact stations passing through for each Public transit users.
- default_commuting_contacts.csv The default number of interacted people if there is no data for the link/mode.
- default_prevalence_rate.pickle prevalence rate for all zip code for one day. Not useful now becasue we use the input from prevalence model
- link_travel_time_distribution.pickle link travel time distribution. A dict {['route','direction','stop_id','time_id']:[samples of link travel time]}
- crowding_distribution.pickle number of people in a vehicle distribution. A dict {['route','direction','stop_id','time_id']:[samples of number of passengers in a vehicle]}
